'''
@File    :   train_unified_encoder_highlevel_diffprior.py
@Time    :   2025/07/13 15:39:04
@Author  :   DongyangLi
@Version :   1.0
@Desc    :   modified from [PAPER_NAME](https://arxiv.org/abs/XXXX.XXXXX) (CONFERENCE_ABBR'YY)
'''


import os
import warnings
import datetime
import argparse
import random
import re
import csv
import time
import math
import itertools
import functools
import multiprocessing
from pathlib import Path
from itertools import combinations

# Deep learning and scientific computing imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn as cudnn

import argparse
import datetime
from torch.utils.data import DataLoader
# Scientific computing libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

# Third-party ML libraries
from einops.layers.torch import Rearrange, Reduce
from adabelief_pytorch import AdaBelief
import tqdm

# WandB for experiment tracking
import wandb
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"
wandb.init(mode="disabled")

# Custom imports for data processing
from data_preparing.eegdatasets import EEGDataset
from data_preparing.megdatasets_averaged import MEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.datasets_mixer import MetaEEGDataset, MetaMEGDataset, MetafMRIDataset, MetaDataLoader

# Custom model components
from model.unified_encoder_multi_tower import UnifiedEncoder
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds

# Custom loss functions and utilities
from loss import ClipLoss, mixco_nce, soft_clip_loss, mixco_1d
from util import wandb_logger
import utils.misc as misc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# System configuration
import sys


def extract_id_from_string(s):
    """
    Extract numeric ID from the end of a string.
    
    Args:
        s (str): Input string containing numeric ID at the end
        
    Returns:
        int or None: Extracted numeric ID, or None if not found
    """
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss implementation.
    
    This loss function is designed to bring samples from the same class closer
    while pushing samples from different classes apart in the embedding space.
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        Initialize SupConLoss.
        
        Args:
            temperature (float): Temperature parameter for scaling similarities
            contrast_mode (str): Mode for contrastive learning ('all' or 'one')
            base_temperature (float): Base temperature for normalization
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Forward pass for SupConLoss.
        
        Args:
            features (torch.Tensor): Feature tensor with shape [batch_size, n_views, ...]
            labels (torch.Tensor, optional): Labels for supervised learning
            mask (torch.Tensor, optional): Mask for positive pairs
            
        Returns:
            torch.Tensor: Computed contrastive loss
        """
        device = features.device

        # Validate input tensor dimensions
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...] tensor')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Normalize features to prevent numerical instability
        features = F.normalize(features, dim=2, eps=1e-8)

        # Handle mask and labels
        if labels is not None and mask is not None:
            raise ValueError('Cannot specify both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Prepare contrast features
        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Invalid `contrast_mode`')

        # Compute similarity matrix
        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-20, max=20)  # Prevent overflow

        # Generate label masks
        mask = mask.repeat(anchor_count, contrast_count)

        # Exclude diagonal elements (self-similarity)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # Compute mean positive log probability
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Compute final loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def train_model(epoch, unified_model, high_pipe, dataloader, optimizer, device, 
                text_features_all, img_features_all, config, eval_modality='eeg'):
    """
    Train the unified model for one epoch.
    
    Args:
        epoch (int): Current epoch number
        unified_model: The unified encoder model
        high_pipe: Diffusion prior pipeline
        dataloader: Training data loader
        optimizer: Optimizer for training
        device: Device to run training on
        text_features_all: All text features
        img_features_all: All image features
        config: Configuration object
        eval_modality (str): Modality for evaluation
        
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
    
    # Initialize training metrics
    total_loss = 0
    correct = 0
    total = 0
    
    # Loss function configurations
    loss_func = ClipLoss()
    supcon_loss_func = SupConLoss()
    
    # Training hyperparameters
    mixup_pct = 0.1
    prior_pct = 0.33
    mse_pct = 0.6
    clip_scale = 1
    mse_scale = 30
    prior_scale = 30
    
    # Loss functions
    mse_loss_fn = nn.MSELoss(reduction='mean')
    prior_criterion = nn.MSELoss(reduction='mean')
    
    # Initialize loss accumulators
    total_regress_loss = 0
    total_loss_clip = 0
    total_prior_loss = 0

    # Training loop
    for batch_idx, (modal, data, labels, text, text_features, img, img_features, 
                    index, img_index, sub_ids) in enumerate(dataloader):
        
        # Move data to device
        data = data.to(device).float()
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Extract subject IDs
        batch_size = data.size(0)
        subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
        subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
        
        # Forward pass through unified model
        neural_features = unified_model(data, subject_ids, modal=modal[0])
        
        # Regression loss (MSE between neural features and image features)
        regress_loss = mse_loss_fn(neural_features, img_features)
        regress_loss *= mse_scale
        total_regress_loss += regress_loss.item()

        # Diffusion prior loss (only after certain epochs)
        prior_loss = 0
        if config.use_prior and epoch > int(prior_pct * config.epochs):
            num_train_timesteps = high_pipe.scheduler.config.num_train_timesteps
            c_embeds = neural_features
            h_embeds = img_features
            N = h_embeds.shape[0]

            # Randomly replace c_embeds with None for robustness
            if torch.rand(1) < 0.1:
                c_embeds = None

            # Generate noise for diffusion process
            noise = torch.randn_like(h_embeds)
            
            # Sample random timesteps
            timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

            # Add noise to embeddings
            perturbed_h_embeds = high_pipe.scheduler.add_noise(h_embeds, noise, timesteps)

            # Predict noise using diffusion prior
            noise_pre = high_pipe.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
            
            # Compute prior loss
            prior_loss = prior_criterion(noise_pre, noise)
            prior_loss *= prior_scale
            total_prior_loss += prior_loss.item()

        # Contrastive loss computation
        logit_scale = unified_model.logit_scale.float()
        
        # Apply mixup augmentation
        neural_features_clone, perm, betas, select = mixco_1d(neural_features.clone())
        
        # Normalize features
        neural_features_norm = nn.functional.normalize(neural_features_clone.flatten(1), dim=-1)
        img_features_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)
        
        # Choose loss based on epoch
        if epoch < int(mixup_pct * config.epochs):
            # Use mixup contrastive loss for early epochs
            loss_clip = mixco_nce(
                neural_features_norm,
                img_features_norm,
                temp=.006,
                perm=perm, betas=betas, select=select)
        else:
            # Use soft contrastive loss for later epochs
            loss_clip = soft_clip_loss(
                neural_features_norm,
                img_features_norm,
                temp=logit_scale)
        
        loss_clip *= clip_scale
        total_loss_clip += loss_clip.item()

        # Combine all losses
        loss = loss_clip + regress_loss + prior_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(unified_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(high_pipe.diffusion_prior.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        high_pipe.diffusion_prior.lr_scheduler.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        logits_img = logit_scale * neural_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        
        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

        # Clean up memory
        del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    # Calculate average metrics
    average_loss = total_loss / (batch_idx + 1)
    average_regress_loss = total_regress_loss / (batch_idx + 1)
    average_loss_clip = total_loss_clip / (batch_idx + 1)
    average_prior_loss = total_prior_loss / (batch_idx + 1)
    accuracy = correct / total

    # Print training statistics
    print(f"Epoch {epoch + 1} - Losses: Regression Loss: {average_regress_loss:.4f}, "
          f"Clip Loss: {average_loss_clip:.4f}, Prior Loss: {average_prior_loss:.4f}")
    
    return average_loss, accuracy


def evaluate_model(epoch, unified_model, high_pipe, dataloader, device, text_features_all, 
                  img_features_all, k, config, eval_modality='eeg'):
    """
    Evaluate the unified model on test data.
    
    Args:
        epoch (int): Current epoch number
        unified_model: The unified encoder model
        high_pipe: Diffusion prior pipeline
        dataloader: Test data loader
        device: Device to run evaluation on
        text_features_all: All text features
        img_features_all: All image features
        k (int): Number of classes for k-way classification
        config: Configuration object
        eval_modality (str): Modality for evaluation
        
    Returns:
        tuple: (average_loss, accuracy, top5_accuracy)
    """
    unified_model.eval()
    
    # Prepare features based on modality
    text_features_all = text_features_all[eval_modality].to(device).float()
    
    if eval_modality == 'eeg' or eval_modality == 'fmri':
        img_features_all = img_features_all[eval_modality].to(device).float()
    elif eval_modality == 'meg':
        img_features_all = img_features_all[eval_modality][::12].to(device).float()
    
    # Initialize evaluation metrics
    total_loss = 0
    correct = 0
    total = 0
    top5_correct_count = 0
    
    # Get all unique classes
    all_labels = set(range(text_features_all.size(0)))
    
    # Loss function configurations
    loss_func = ClipLoss()
    supcon_loss_func = SupConLoss()
    mixup_pct = 0.33
    clip_scale = 1
    batch_idx = 0

    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, 
                        index, img_index, sub_ids) in enumerate(dataloader):
            
            # Move data to device
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            # Extract subject IDs
            batch_size = data.size(0)
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Forward pass
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            
            logit_scale = unified_model.logit_scale.float()

            # Apply mixup augmentation
            neural_features, perm, betas, select = mixco_1d(neural_features)
            neural_features_norm = nn.functional.normalize(neural_features.flatten(1), dim=-1)
            img_features_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)
            
            # Choose loss based on epoch
            if epoch < int(mixup_pct * config.epochs):
                loss_clip = mixco_nce(
                    neural_features_norm,
                    img_features_norm,
                    temp=.006,
                    perm=perm, betas=betas, select=select)
            else:
                loss_clip = soft_clip_loss(
                    neural_features_norm,
                    img_features_norm,
                    temp=logit_scale)
            
            loss_clip *= clip_scale
            loss = loss_clip
            total_loss += loss.item()
            
            # Evaluate each sample in the batch
            for idx, label in enumerate(labels):
                # Select k-1 random classes plus the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                if k == 200:
                    # Full 200-class evaluation
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    predicted_label = selected_classes[torch.argmax(logits_img).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    
                    # Top-5 accuracy
                    _, top5_indices = torch.topk(logits_img, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    total += 1
                    
                elif k == 50 or k == 100:
                    # k-way classification for k=50 or k=100
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    predicted_label = selected_classes[torch.argmax(logits_img).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    
                    # Top-5 accuracy
                    _, top5_indices = torch.topk(logits_img, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    total += 1
                    
                elif k in [2, 4, 10]:
                    # k-way classification for smaller k
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    predicted_label = selected_classes[torch.argmax(logits_img).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error: Invalid k value.")
                    
            # Clean up memory
            del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    # Calculate average metrics
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    
    return average_loss, accuracy, top5_acc


def main_train_loop(test_subjects, current_time, unified_model, high_pipe, train_dataloader, 
                   test_dataloader, optimizer, device, text_features_train_all, 
                   text_features_test_all, img_features_train_all, img_features_test_all, 
                   config, logger=None, eval_modality='eeg'):
    """
    Main training loop for the unified model.
    
    Args:
        test_subjects: List of test subject IDs
        current_time: Current timestamp for saving models
        unified_model: The unified encoder model
        high_pipe: Diffusion prior pipeline
        train_dataloader: Training data loader
        test_dataloader: Test data loader
        optimizer: Optimizer for training
        device: Device to run training on
        text_features_train_all: All training text features
        text_features_test_all: All test text features
        img_features_train_all: All training image features
        img_features_test_all: All test image features
        config: Configuration object
        logger: Logger for experiment tracking
        eval_modality (str): Modality for evaluation
        
    Returns:
        list: Results for each epoch
    """
    # Initialize logger
    logger = wandb_logger(config) if logger else None
    logger.watch(unified_model, logger)
    
    # Initialize metrics tracking
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs, v4_accs, v10_accs = [], [], []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Setup diffusion prior scheduler if using prior
    if config.use_prior:
        high_pipe.diffusion_prior.train()
        from diffusers.optimization import get_cosine_schedule_with_warmup
        high_pipe.diffusion_prior.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(train_dataloader) * config.epochs),
        )
    
    # Main training loop
    for epoch in range(config.epochs):
        # Training phase
        train_loss, train_accuracy = train_model(
            epoch, unified_model, high_pipe, train_dataloader, optimizer, device,
            text_features_train_all, img_features_train_all, config=config, 
            eval_modality=eval_modality
        )
        
        # Save model checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Create directories for saving models
            os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)
            file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
            torch.save(unified_model.state_dict(), file_path)
            
            # Save diffusion prior model
            os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}/prior_diffusion", exist_ok=True)
            prior_file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/prior_diffusion/{epoch+1}.pth"
            torch.save(high_pipe.diffusion_prior.state_dict(), prior_file_path)
            
            print(f"Unified model saved in {file_path}!")
            print(f"Prior diffusion model saved in {prior_file_path}!")
        
        # Record training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation phase
        if eval_modality == 'fmri':
            test_loss, test_accuracy, top5_acc = evaluate_model(
                epoch, unified_model, high_pipe, test_dataloader, device,
                text_features_test_all, img_features_test_all, k=100, config=config,
                eval_modality=eval_modality
            )
        else:
            test_loss, test_accuracy, top5_acc = evaluate_model(
                epoch, unified_model, high_pipe, test_dataloader, device,
                text_features_test_all, img_features_test_all, k=200, config=config,
                eval_modality=eval_modality
            )
        
        # Evaluate on different k-way classification tasks
        _, v2_acc, _ = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device,
            text_features_test_all, img_features_test_all, k=2, config=config,
            eval_modality=eval_modality
        )
        _, v4_acc, _ = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device,
            text_features_test_all, img_features_test_all, k=4, config=config,
            eval_modality=eval_modality
        )
        _, v10_acc, _ = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device,
            text_features_test_all, img_features_test_all, k=10, config=config,
            eval_modality=eval_modality
        )
        _, v50_acc, v50_top5_acc = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device,
            text_features_test_all, img_features_test_all, k=50, config=config,
            eval_modality=eval_modality
        )
        _, v100_acc, v100_top5_acc = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device,
            text_features_test_all, img_features_test_all, k=100, config=config,
            eval_modality=eval_modality
        )
        
        # Record test metrics
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
        
        # Track best model
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
        
        # Log metrics to wandb
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

        # Print epoch results
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"v2 Accuracy: {v2_acc} - v4 Accuracy: {v4_acc} - v10 Accuracy: {v10_acc} - "
              f"v50 Accuracy: {v50_acc} - v100 Accuracy: {v100_acc}")

    # Create visualization plots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # k-way accuracy plots
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def create_optimizer_for_multiple_models(models, use_prior=True, max_lr=1e-3):
    """
    Create an optimizer for multiple models with grouped parameters.
    
    Args:
        models: List of models to optimize
        use_prior: Whether to use diffusion prior (default: True)
        max_lr: Maximum learning rate (default: 1e-3)
    
    Returns:
        AdamW optimizer configured for all models
    """
    opt_grouped_parameters = []

    # Iterate through all models and add their optimizer parameters
    for model in models:
        if hasattr(model, 'diffusion_prior') and use_prior:
            opt_grouped_parameters.append({
                'params': model.parameters()
            })
        else:
            # Add all model parameters to the optimizer
            opt_grouped_parameters.append({
                'params': model.parameters()
            })

    # Create the final optimizer
    optimizer = AdamW(opt_grouped_parameters, lr=max_lr)
    return optimizer


def main():
    """
    Main function to handle argument parsing and training setup for multi-modal neural decoding.
    Supports EEG, MEG, and fMRI modalities with unified encoder architecture.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    
    # Pre-trained encoder paths for different modalities
    parser.add_argument(
        '--encoder_paths',
        nargs='+',
        required=False,
        default=[
            'eeg=./checkpoints/eeg_encoder.pth',
            'meg=./checkpoints/meg_encoder.pth',
            'fmri=./checkpoints/fmri_encoder.pth'
        ],
        help='Paths to pre-trained encoders for each modality (modify according to your checkpoint location)'
    )
    
    # Modality selection parameters
    parser.add_argument(
        '--modalities', 
        nargs='+', 
        choices=['eeg', 'meg', 'fmri'], 
        default=['eeg', 'meg', 'fmri'], 
        help='List of modalities to train on (e.g., eeg, meg, fmri)'
    )
    
    parser.add_argument(
        '--eval_modality', 
        type=str, 
        choices=['eeg', 'meg', 'fmri'], 
        default='fmri', 
        help='Modality to evaluate on'
    )
    
    # Model architecture parameters
    parser.add_argument(
        '--depth', 
        type=int, 
        default=2, 
        help='Depth of the model (default: 2)'
    )
    
    # Dataset paths for different modalities
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
    
    # Output and logging configuration
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
    
    # Training hyperparameters
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
        default=250, 
        help='Batch size'
    )
    
    # System configuration
    parser.add_argument(
        '--logger', 
        type=bool, 
        default=True, 
        help='Enable WandB logging'
    )
    
    parser.add_argument(
        '--gpu', 
        type=str, 
        default='cuda:6', 
        help='GPU device to use'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cpu', 'gpu'], 
        default='gpu', 
        help='Device to run on (cpu or gpu)'
    )
    
    # Experimental setup
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
    
    # Subject selection for different modalities
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
    
    # Diffusion prior configuration
    parser.add_argument(
        "--use_prior", 
        action=argparse.BooleanOptionalAction, 
        default=True, 
        help="Whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)"
    )
    
    args = parser.parse_args()
    
    # Parse encoder paths from command line arguments
    encoder_paths = {}
    for path in args.encoder_paths:
        key, value = path.split('=')
        encoder_paths[key] = value
    
    # Set device based on the argument
    device = torch.device(args.gpu if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')

    # Initialize empty datasets and feature containers for each modality
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

    # Initialize training setup
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    
    # Create unified encoder model
    unified_model = UnifiedEncoder(
        encoder_paths, 
        device, 
        num_experts=5, 
        num_heads=args.depth, 
        ff_dim=64 * args.depth, 
        num_layers=args.depth
    )
    unified_model.to(device)

    # Create diffusion prior model
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    high_pipe = Pipe(diffusion_prior, device=device)
    high_pipe.to(device)
    
    # Combine models for optimization
    models = [unified_model, diffusion_prior]
    optimizer = create_optimizer_for_multiple_models(models, use_prior=True, max_lr=1e-3)

    # Alternative optimizer option (AdaBelief) - currently commented out
    # optimizer = AdaBelief(
    #     itertools.chain(unified_model.parameters()),
    #     lr=args.lr,
    #     betas=(0.9, 0.999),  # Same beta parameters as AdamW
    #     eps=1e-16,           # Same epsilon parameter as AdamW
    #     weight_decay=1e-2,   # Same weight decay parameter as AdamW
    #     rectify=False        # Set to True if you want to use Rectified AdaBelief
    # )
    
    # Print parameter information for debugging
    for name, param in unified_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    def format_num(num):
        """Format large numbers with appropriate units (K, M, B, T, P)"""
        for unit in ['', 'K', 'M', 'B', 'T']:
            if num < 1000:
                return f"{num:.2f}{unit}"
            num /= 1000
        return f"{num:.2f}P"

    # Calculate and print total and trainable parameters
    total_params = sum(p.numel() for p in unified_model.parameters())
    trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")

    # Calculate and print trainable parameters percentage
    if total_params > 0:
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
    else:
        print("Total parameters count is zero, cannot compute percentage.")

    # Create meta data loader dynamically based on selected modalities
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

    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )

    # Collect test features for evaluation
    text_features_test_all[args.eval_modality] = test_dataset.text_features
    img_features_test_all[args.eval_modality] = test_dataset.img_features

    # Execute the main training loop
    results = main_train_loop(
        args.test_subjects, 
        current_time, 
        unified_model, 
        high_pipe, 
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


if __name__ == '__main__':
    main()