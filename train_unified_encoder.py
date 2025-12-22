'''
@File    :   train_unified_encoder.py
@Time    :   2025/07/13 14:55:06
@Author  :   DongyangLi
@Version :   1.0
@Desc    :   modified from [PAPER_NAME](https://arxiv.org/abs/XXXX.XXXXX) (CONFERENCE_ABBR'YY)
'''


import argparse
import copy
import csv
import datetime
import itertools
import math
import multiprocessing
import os
import random
import re
import time
import warnings
from itertools import combinations
from pathlib import Path
import functools

import argparse
import os
import datetime
import torch
from torch.utils.data import DataLoader
import itertools
from transformers import AdamW
import csv
from adabelief_pytorch import AdaBelief
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
import wandb
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Local imports
from data_preparing.datasets_mixer import MetaEEGDataset, MetaMEGDataset, MetafMRIDataset, MetaDataLoader
from data_preparing.eegdatasets import EEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.megdatasets_averaged import MEGDataset
from loss import ClipLoss
from model.unified_encoder_multi_tower import UnifiedEncoder
from util import wandb_logger
import utils.misc as misc

# Environment variables
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"

# Configure warnings and wandb
warnings.filterwarnings("ignore")
wandb.init(mode="disabled")

# Commented out imports
# from data.pretrain_dataset import PretrainDataset, ConcatDataset



def extract_id_from_string(s):
    """Extract numeric ID from the end of a string.
    
    Args:
        s (str): Input string
        
    Returns:
        int or None: Extracted ID as integer, None if not found
    """
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss function.
    
    Reference: https://arxiv.org/abs/2004.11362
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """Initialize SupConLoss.
        
        Args:
            temperature (float): Temperature parameter for scaling similarities
            contrast_mode (str): Contrast mode, either 'all' or 'one'
            base_temperature (float): Base temperature for loss scaling
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute supervised contrastive loss.
        
        Args:
            features (torch.Tensor): Feature tensor of shape [batch_size, n_views, feature_dim]
            labels (torch.Tensor, optional): Label tensor of shape [batch_size]
            mask (torch.Tensor, optional): Mask tensor for positive pairs
            
        Returns:
            torch.Tensor: Computed loss value
        """
        device = features.device
        
        # Validate input tensor shape
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be a tensor of shape [batch_size, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Normalize features to unit vectors
        features = F.normalize(features, dim=2, eps=1e-8)  # Add eps to avoid normalizing zero vectors

        # Handle labels and mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot specify both `labels` and `mask`')
        elif labels is None and mask is None:
            # Use identity matrix as mask (each sample is positive with itself)
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Create mask from labels (samples with same label are positive pairs)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Set up contrast features and anchors
        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            # Use only first view as anchor
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # Use all views as anchors
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Invalid `contrast_mode`')

        # Compute similarity matrix
        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-20, max=20)  # Clamp to prevent overflow

        # Expand mask for all anchor-contrast pairs
        mask = mask.repeat(anchor_count, contrast_count)

        # Create logits mask (exclude diagonal elements as they are not valid positive pairs)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)  # Avoid log(0)

        # Compute mean log probability for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)  # Avoid division by zero

        # Compute final loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
def train_model(unified_model, dataloader, optimizer, device, text_features_all, 
                img_features_all, config, eval_modality='eeg'):
    """
    Train the unified model for one epoch
    
    Args:
        unified_model: The unified encoder model
        dataloader: Training data loader
        optimizer: Optimizer for training
        device: Device to run on
        text_features_all: Dictionary of text features for all modalities
        img_features_all: Dictionary of image features for all modalities
        config: Configuration object
        eval_modality (str): Modality to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    unified_model.train()
    
    # Prepare features based on modality
    if eval_modality == 'eeg':
        img_features_all = img_features_all[eval_modality][::10].to(device).float()
    elif eval_modality in ['meg', 'fmri']:
        img_features_all = img_features_all[eval_modality][::12].to(device).float()
        
    text_features_all = text_features_all[eval_modality].to(device).float()
    
    # Initialize metrics
    total_loss = 0
    correct = 0
    total = 0
    
    # Initialize loss functions
    loss_func = ClipLoss()
    supcon_loss_func = SupConLoss()
    
    # Ensure all features are in float32
    img_features_all = img_features_all.float()
    text_features_all = text_features_all.float()
    
    for batch_idx, batch_data in enumerate(dataloader):
        (modal, data, labels, text, text_features, img, 
         img_features, index, img_index, sub_ids) = batch_data
        
        # Move data to device
        data = data.to(device).float()
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Extract subject IDs
        subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
        subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
        
        # Forward pass
        neural_features = unified_model(data, subject_ids, modal=modal[0])
        logit_scale = unified_model.logit_scale.float()
        
        # Compute losses
        img_loss = loss_func(neural_features, img_features, logit_scale)
        text_loss = loss_func(neural_features, text_features, logit_scale)
        loss = img_loss
        
        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(unified_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        logits_img = logit_scale * neural_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        
        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        
        # Clean up memory
        del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy


def evaluate_model(unified_model, dataloader, device, text_features_all, 
                   img_features_all, k, config, eval_modality='eeg'):
    """
    Evaluate the unified model
    
    Args:
        unified_model: The unified encoder model
        dataloader: Evaluation data loader
        device: Device to run on
        text_features_all: Dictionary of text features for all modalities
        img_features_all: Dictionary of image features for all modalities
        k (int): Number of classes for k-way classification
        config: Configuration object
        eval_modality (str): Modality to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy, top5_accuracy)
    """
    unified_model.eval()
    
    # Prepare features
    text_features_all = text_features_all[eval_modality].to(device).float()
    if eval_modality in ['eeg', 'fmri']:
        img_features_all = img_features_all[eval_modality].to(device).float()
    elif eval_modality == 'meg':
        img_features_all = img_features_all[eval_modality][::12].to(device).float()
    
    # Initialize metrics
    total_loss = 0
    correct = 0
    total = 0
    top5_correct_count = 0
    
    # Initialize loss functions
    loss_func = ClipLoss()
    supcon_loss_func = SupConLoss()
    
    img_features_all = img_features_all.float()
    text_features_all = text_features_all.float()
    
    # Get all unique classes
    all_labels = set(range(text_features_all.size(0)))
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            (modal, data, labels, text, text_features, img, 
             img_features, index, img_index, sub_ids) = batch_data
            
            # Move data to device
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            # Extract subject IDs
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Forward pass
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            logit_scale = unified_model.logit_scale.float()
            
            # Compute losses
            img_loss = loss_func(neural_features, img_features, logit_scale)
            text_loss = loss_func(neural_features, text_features, logit_scale)
            loss = img_loss
            
            total_loss += loss.item()
            
            # Evaluate for each sample in the batch
            for idx, label in enumerate(labels):
                # Select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                # Compute logits
                logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                logits_single = logits_img
                
                # Get predicted class
                predicted_label = selected_classes[torch.argmax(logits_single).item()]
                if predicted_label == label.item():
                    correct += 1
                
                # Compute top-5 accuracy for larger k values
                if k >= 5:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                
                total += 1
            
            # Clean up memory
            del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total if total > 0 else 0
    
    return average_loss, accuracy, top5_acc

def main_train_loop(test_subjects, current_time, unified_model, train_dataloader, test_dataloader, 
                   optimizer, device, text_features_train_all, text_features_test_all, 
                   img_features_train_all, img_features_test_all, config, logger=None, eval_modality='eeg'):
    """
    Main training loop for multimodal neural signal processing.
    
    This function orchestrates the complete training process including model training,
    evaluation, checkpointing, and result visualization for EEG/MEG/fMRI data.
    
    Args:
        test_subjects (list): List of test subject IDs
        current_time (str): Current timestamp for model saving
        unified_model: The unified encoder model to train
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        optimizer: Optimizer for model training
        device: Device to run training on (CPU/GPU)
        text_features_train_all (dict): Training text features for all modalities
        text_features_test_all (dict): Test text features for all modalities
        img_features_train_all (dict): Training image features for all modalities
        img_features_test_all (dict): Test image features for all modalities
        config: Configuration object with training parameters
        logger (bool, optional): Whether to use WandB logging. Defaults to None.
        eval_modality (str, optional): Modality to evaluate on. Defaults to 'eeg'.
    
    Returns:
        list: Results for each epoch containing accuracy metrics
    """
    
    # Initialize logger if enabled
    logger = wandb_logger(config) if logger else None
    if logger:
        logger.watch(unified_model, logger)
    
    # Initialize tracking lists for metrics
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs, v4_accs, v10_accs = [], [], []
    
    # Best model tracking
    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # Store results for each epoch
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Main training loop
    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        
        # Training phase
        train_loss, train_accuracy = train_model(
            unified_model, 
            train_dataloader, 
            optimizer, 
            device, 
            text_features_train_all, 
            img_features_train_all, 
            config=config, 
            eval_modality=eval_modality
        )
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f"./models/FloraHard/across/{config.encoder_type}/{current_time}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/{epoch + 1}.pth"
            torch.save(unified_model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at: {checkpoint_path}")
        
        # Store training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation phase with different k values
        print("Evaluating model performance...")
        
        # Main evaluation with different k values based on modality
        if eval_modality == 'fmri':
            test_loss, test_accuracy, top5_acc = evaluate_model(
                unified_model, test_dataloader, device, text_features_test_all, 
                img_features_test_all, k=100, config=config, eval_modality=eval_modality
            )
        else:
            test_loss, test_accuracy, top5_acc = evaluate_model(
                unified_model, test_dataloader, device, text_features_test_all, 
                img_features_test_all, k=200, config=config, eval_modality=eval_modality
            )
        
        # Evaluate with different k values for comprehensive analysis
        _, v2_acc, _ = evaluate_model(
            unified_model, test_dataloader, device, text_features_test_all, 
            img_features_test_all, k=2, config=config, eval_modality=eval_modality
        )
        
        _, v4_acc, _ = evaluate_model(
            unified_model, test_dataloader, device, text_features_test_all, 
            img_features_test_all, k=4, config=config, eval_modality=eval_modality
        )
        
        _, v10_acc, _ = evaluate_model(
            unified_model, test_dataloader, device, text_features_test_all, 
            img_features_test_all, k=10, config=config, eval_modality=eval_modality
        )
        
        _, v50_acc, v50_top5_acc = evaluate_model(
            unified_model, test_dataloader, device, text_features_test_all, 
            img_features_test_all, k=50, config=config, eval_modality=eval_modality
        )
        
        _, v100_acc, v100_top5_acc = evaluate_model(
            unified_model, test_dataloader, device, text_features_test_all, 
            img_features_test_all, k=100, config=config, eval_modality=eval_modality
        )
        
        # Store evaluation metrics
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Store epoch results
        epoch_results = {
            "epoch": epoch + 1,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "v2_acc": v2_acc,
            "v4_acc": v4_acc,
            "v10_acc": v10_acc,
            "top5_acc": top5_acc,
            "v50_acc": v50_acc,
            "v100_acc": v100_acc,
            "v50_top5_acc": v50_top5_acc,
            "v100_top5_acc": v100_top5_acc
        }
        results.append(epoch_results)
        
        # Track best model performance
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc": v2_acc,
                "v4_acc": v4_acc,
                "v10_acc": v10_acc
            }
            print(f"New best model found! Test accuracy: {best_accuracy:.4f}")
        
        # Log metrics to WandB
        if logger:
            logger.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy,
                "v2 Accuracy": v2_acc,
                "v4 Accuracy": v4_acc,
                "v10 Accuracy": v10_acc,
                "Epoch": epoch
            })
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"v2 Accuracy: {v2_acc:.4f}, v4 Accuracy: {v4_acc:.4f}, v10 Accuracy: {v10_acc:.4f}")
        print(f"v50 Accuracy: {v50_acc:.4f}, v100 Accuracy: {v100_acc:.4f}")
    
    # Generate training visualization
    print("\nGenerating training visualization...")
    _generate_training_plots(
        train_losses, test_losses, train_accuracies, test_accuracies,
        v2_accs, v4_accs, v10_accs, best_epoch_info
    )
    
    # Finalize logging
    if logger:
        logger.finish()
    
    print(f"\nTraining completed! Best test accuracy: {best_accuracy:.4f} at epoch {best_epoch_info['epoch']}")
    return results


def _generate_training_plots(train_losses, test_losses, train_accuracies, test_accuracies,
                           v2_accs, v4_accs, v10_accs, best_epoch_info):
    """
    Generate comprehensive training visualization plots.
    
    Args:
        train_losses (list): Training loss values
        test_losses (list): Test loss values
        train_accuracies (list): Training accuracy values
        test_accuracies (list): Test accuracy values
        v2_accs (list): 2-class accuracy values
        v4_accs (list): 4-class accuracy values
        v10_accs (list): 10-class accuracy values
        best_epoch_info (dict): Information about the best performing epoch
    """
    
    # Create subplot grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    
    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axs[0, 0].plot(test_losses, label='Test Loss', linewidth=2)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Training and Test Loss")
    axs[0, 0].grid(True, alpha=0.3)
    
    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy', linewidth=2)
    axs[0, 1].plot(test_accuracies, label='Test Accuracy', linewidth=2)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Training and Test Accuracy")
    axs[0, 1].grid(True, alpha=0.3)
    
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy', linewidth=2, color='orange')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy")
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy', linewidth=2, color='green')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy")
    axs[1, 1].grid(True, alpha=0.3)
    
    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy', linewidth=2, color='red')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy")
    axs[2, 0].grid(True, alpha=0.3)
    
    # Best model information panel
    info_text = (
        f"Best Model Performance (Epoch {best_epoch_info['epoch']}):\n"
        f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
        f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
        f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
        f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
        f"2-class Accuracy: {best_epoch_info['v2_acc']:.4f}\n"
        f"4-class Accuracy: {best_epoch_info['v4_acc']:.4f}\n"
        f"10-class Accuracy: {best_epoch_info['v10_acc']:.4f}"
    )
    
    axs[2, 1].axis('off')
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=11, ha='center', va='center', 
                   transform=axs[2, 1].transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightblue", alpha=0.7))
    
    # Layout adjustment and save
    plt.tight_layout()
    plt.suptitle('Multimodal Neural Signal Training Results', fontsize=16, y=0.98)
    
    # Save plot
    plot_filename = f'training_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved as: {plot_filename}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Main function for EEG/MEG/fMRI Transformer Training Script.
    
    This function sets up the training pipeline for multimodal neural signal processing,
    supporting EEG, MEG, and fMRI data modalities with unified encoder architecture.
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    
    # Encoder model paths for different modalities
    parser.add_argument(
        '--encoder_paths',
        nargs='+',
        required=False,
        default=[
            'eeg=./checkpoints/eeg_encoder.pth',
            'meg=./checkpoints/meg_encoder.pth',
            'fmri=./checkpoints/fmri_encoder.pth'
        ],
        help='Paths to pre-trained encoder models for each modality (modify according to your checkpoint location)'
    )
    
    # Modality selection for training
    parser.add_argument(
        '--modalities', 
        nargs='+', 
        choices=['eeg', 'meg', 'fmri'], 
        default=['eeg', 'meg', 'fmri'], 
        help='List of modalities to train on (e.g., eeg, meg, fmri)'
    )
    
    # Evaluation modality specification
    parser.add_argument(
        '--eval_modality', 
        type=str, 
        choices=['eeg', 'meg', 'fmri'], 
        default='fmri', 
        help='Modality to evaluate on'
    )
    
    # Dataset paths
    parser.add_argument(
        '--eeg_data_path', 
        type=str, 
        default="./data/THINGS_EEG/Preprocessed_data_250Hz", 
        help='Path to the EEG dataset (modify according to your dataset location)'
    )
    parser.add_argument(
        '--meg_data_path', 
        type=str, 
        default="./data/THINGS_MEG/preprocessed_newsplit", 
        help='Path to the MEG dataset (modify according to your dataset location)'
    )
    parser.add_argument(
        '--fmri_data_path', 
        type=str, 
        default="./data/fmri_dataset/Preprocessed", 
        help='Path to the fMRI dataset (modify according to your dataset location)'
    )
    
    # Training configuration
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./outputs/contrast', 
        help='Directory to save output results'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default="train_pos_img_text_rep", 
        help='WandB project name'
    )
    parser.add_argument(
        '--entity', 
        type=str, 
        default="sustech_rethinkingbci", 
        help='WandB entity name'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default="lr=3e-4_img_pos_pro_eeg", 
        help='Experiment name'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=3e-4, 
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=150, 
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=300, 
        help='Batch size'
    )
    parser.add_argument(
        '--logger', 
        type=bool, 
        default=True, 
        help='Enable WandB logging'
    )
    parser.add_argument(
        '--gpu', 
        type=str, 
        default='cuda:2', 
        help='GPU device to use'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cpu', 'gpu'], 
        default='gpu', 
        help='Device to run on (cpu or gpu)'
    )
    
    # Subject configuration
    parser.add_argument(
        '--insubject', 
        type=bool, 
        default=True, 
        help='In-subject mode or cross-subject mode'
    )
    parser.add_argument(
        '--encoder_type', 
        type=str, 
        default='Unified_EEG+MEG+fMRI_EEG', 
        help='Encoder type'
    )
    parser.add_argument(
        '--test_subjects', 
        nargs='+', 
        default=['sub-02'], 
        help='Subject ID to test on'
    )
    parser.add_argument(
        '--eeg_subjects', 
        nargs='+', 
        default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
                'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], 
        help='List of EEG subject IDs (default: sub-01 to sub-10)'
    )
    parser.add_argument(
        '--meg_subjects', 
        nargs='+', 
        default=['sub-01', 'sub-02', 'sub-03', 'sub-04'], 
        help='List of MEG subject IDs'
    )
    parser.add_argument(
        '--fmri_subjects', 
        nargs='+', 
        default=['sub-01', 'sub-02', 'sub-03'], 
        help='List of fMRI subject IDs'
    )
    
    args = parser.parse_args()
    
    # Parse encoder paths from command line arguments
    encoder_paths = {}
    for path in args.encoder_paths:
        key, value = path.split('=')
        encoder_paths[key] = value
    
    # Set device based on the argument
    device = torch.device(args.gpu if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    
    # Initialize empty datasets for each modality
    eeg_train_dataset = None
    meg_train_dataset = None
    fmri_train_dataset = None
    text_features_train_all = {}
    text_features_test_all = {}
    img_features_train_all = {}
    img_features_test_all = {}
    
    # Load datasets based on selected modalities
    if 'eeg' in args.modalities:
        eeg_train_dataset = MetaEEGDataset(args.eeg_data_path, args.eeg_subjects, train=True)
        text_features_train_all['eeg'] = eeg_train_dataset.text_features
        img_features_train_all['eeg'] = eeg_train_dataset.img_features
    
    if 'meg' in args.modalities:
        meg_train_dataset = MetaMEGDataset(args.meg_data_path, args.meg_subjects, train=True)
        text_features_train_all['meg'] = meg_train_dataset.text_features
        img_features_train_all['meg'] = meg_train_dataset.img_features
    
    if 'fmri' in args.modalities:
        fmri_train_dataset = MetafMRIDataset(args.fmri_data_path, args.fmri_subjects, train=True)
        text_features_train_all['fmri'] = fmri_train_dataset.text_features
        img_features_train_all['fmri'] = fmri_train_dataset.img_features
    
    # Initialize training components
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    unified_model = UnifiedEncoder(encoder_paths, device)
    unified_model.to(device)
    
    optimizer = AdamW(itertools.chain(unified_model.parameters()), lr=args.lr)
    
    # Print model parameter information
    for name, param in unified_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    def format_num(num):
        """Format large numbers with appropriate unit suffixes."""
        for unit in ['', 'K', 'M', 'B', 'T']:
            if num < 1000:
                return f"{num:.2f}{unit}"
            num /= 1000
        return f"{num:.2f}P"
    
    # Calculate and print model parameter statistics
    total_params = sum(p.numel() for p in unified_model.parameters())
    trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
    print(f"Total parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")
    
    # Calculate and print trainable parameter percentage
    if total_params > 0:
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
    else:
        print("Total parameters count is zero, cannot compute percentage.")
    
    # Define the meta data loader dynamically based on selected modalities
    metadataloader = MetaDataLoader(
        eeg_dataset=eeg_train_dataset if 'eeg' in args.modalities else None,
        meg_dataset=meg_train_dataset if 'meg' in args.modalities else None,
        fmri_dataset=fmri_train_dataset if 'fmri' in args.modalities else None,
        batch_size=args.batch_size,
        drop_last=True,
        modalities=args.modalities
    )
    train_loader = metadataloader
    
    # Prepare test dataset based on eval_modality and test_subjects
    if args.eval_modality == 'eeg':
        test_dataset = EEGDataset(args.eeg_data_path, subjects=args.test_subjects, train=False)
    elif args.eval_modality == 'meg':
        test_dataset = MEGDataset(args.meg_data_path, subjects=args.test_subjects, train=False)
    elif args.eval_modality == 'fmri':
        test_dataset = fMRIDataset(args.fmri_data_path, subjects=args.test_subjects, train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    
    # Collect test features
    text_features_test_all[args.eval_modality] = test_dataset.text_features
    img_features_test_all[args.eval_modality] = test_dataset.img_features
    
    # Perform the main training loop
    results = main_train_loop(
        args.test_subjects, 
        current_time, 
        unified_model, 
        train_loader, 
        test_loader, 
        optimizer, 
        device,
        text_features_train_all, 
        text_features_test_all, 
        img_features_train_all, 
        img_features_test_all, 
        config=args,
        logger=args.logger, 
        eval_modality=args.eval_modality
    )
    
    return results


if __name__ == '__main__':
    main()

