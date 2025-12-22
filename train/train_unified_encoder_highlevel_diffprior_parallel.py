'''
@File    :   train_unified_encoder_highlevel_diffprior_parallel.py
@Time    :   2025/07/13 16:16:47
@Author  :   DongyangLi
@Version :   1.0
@Desc    :   modified from [PAPER_NAME](https://arxiv.org/abs/XXXX.XXXXX) (CONFERENCE_ABBR'YY)

Run from project root: python -m train.train_unified_encoder_highlevel_diffprior_parallel
Or: python train/train_unified_encoder_highlevel_diffprior_parallel.py
'''

import sys
from pathlib import Path
# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import random
import math
import time
import csv
import warnings
import argparse
import datetime
from pathlib import Path
from itertools import combinations
from functools import partial
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms

from einops.layers.torch import Rearrange, Reduce
from transformers import AdamW
from accelerate import Accelerator
import wandb
from adabelief_pytorch import AdaBelief
from diffusers.optimization import get_cosine_schedule_with_warmup

# Custom imports
from data_preparing.eegdatasets import EEGDataset
from data_preparing.megdatasets_averaged import MEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.datasets_mixer import (MetaEEGDataset, MetaMEGDataset, 
                                         MetafMRIDataset, MetaDataLoader)

from utils.losses import ClipLoss, mixco_nce, soft_clip_loss, mixco_1d
from model.unified_encoder_multi_tower import UnifiedEncoder
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds
from utils import wandb_logger
import utils.misc as misc

# Disable warnings and setup environment
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

# Initialize Accelerator
accelerator = Accelerator(device_placement=True, split_batches=True, mixed_precision='bf16')
print = accelerator.print  # Override print to work with distributed training

# Device setup
device = accelerator.device
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
num_workers = num_devices

# Distributed training info
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print(f"distributed = {distributed}, num_devices = {num_devices}, "
      f"local rank = {local_rank}, world size = {world_size}")
print(f"PID of this process = {os.getpid()}")

def extract_id_from_string(s):
    """Extract numerical ID from string ending with digits"""
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_num(num):
    """Format large numbers with appropriate units (K, M, B, etc.)"""
    for unit in ['','K','M','B','T']:
        if num < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"

def create_optimizer_for_multiple_models(models, use_prior=True, max_lr=1e-3):
    """
    Create optimizer for multiple models with optional prior network
    
    Args:
        models: List of models to optimize
        use_prior: Whether to include prior network parameters
        max_lr: Maximum learning rate
        
    Returns:
        AdamW optimizer configured for all models
    """
    opt_grouped_parameters = []
    
    for model in models:
        if hasattr(model, 'diffusion_prior') and use_prior:
            opt_grouped_parameters.append({'params': model.parameters()})
        else:
            opt_grouped_parameters.append({'params': model.parameters()})
            
    return AdamW(opt_grouped_parameters, lr=max_lr)

def train_model(epoch, unified_model, high_pipe, dataloader, optimizer, device, 
               text_features_all, img_features_all, config, accelerator, eval_modality='eeg'):
    """
    Training loop for one epoch
    
    Args:
        epoch: Current epoch number
        unified_model: Main model to train
        high_pipe: Diffusion prior pipeline
        dataloader: Data loader for training
        optimizer: Optimizer
        device: Device to run on
        text_features_all: Precomputed text features
        img_features_all: Precomputed image features  
        config: Configuration dictionary
        accelerator: Accelerator for distributed training
        eval_modality: Which modality to evaluate on
        
    Returns:
        Average loss and accuracy for the epoch
    """
    unified_model.train()    
    
    # Select appropriate features based on modality
    if eval_modality == 'eeg':
        img_features_all = img_features_all[eval_modality][::10].to(device).float()
    elif eval_modality in ['meg', 'fmri']:
        img_features_all = img_features_all[eval_modality][::12].to(device).float()
        
    text_features_all = text_features_all[eval_modality].to(device).float()  # (n_cls, d)
    
    # Initialize metrics
    total_loss = 0
    correct = 0
    total = 0
    loss_func = ClipLoss()
    num_voxels = {1: 6036, 2: 5944, 3: 5238}
    
    # Ensure correct precision
    img_features_all = img_features_all.float()
    text_features_all = text_features_all.float()
    mixup_pct = .33
    clip_scale = 1    
    mse_loss_fn = nn.MSELoss(reduction='mean')
    prior_loss_sum = 0
    prior_criterion = nn.MSELoss(reduction='mean')
    
    # Training loop
    for batch_idx, (modal, data, labels, text, text_features, img, img_features, 
                   index, img_index, sub_ids) in enumerate(dataloader):
        
        # Move data to device
        data = data.to(device).float()
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = data.size(0)
        subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
        subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
        
        # Forward pass
        neural_features = unified_model(data, subject_ids, modal=modal[0])        
        regress_loss = mse_loss_fn(neural_features, img_features)
        
        # Diffusion prior training if enabled
        if config.use_prior:     
            num_train_timesteps = high_pipe.scheduler.config.num_train_timesteps
            c_embeds = neural_features
            h_embeds = img_features
            N = h_embeds.shape[0]

            # 1. Randomly replace c_embeds with None (10% chance)
            if torch.rand(1) < 0.1:
                c_embeds = None

            # 2. Generate noisy embeddings as input
            noise = torch.randn_like(h_embeds)

            # 3. Sample timesteps
            timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

            # 4. Add noise to h_embedding
            perturbed_h_embeds = high_pipe.scheduler.add_noise(
                h_embeds,
                noise,
                timesteps
            )

            # 5. Predict noise
            noise_pre = high_pipe.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
            
            # 6. Loss function weighted by sigma
            prior_loss = prior_criterion(noise_pre, noise)
            prior_loss_sum += prior_loss.item()

        logit_scale = unified_model.logit_scale.float()
        
        # Mixup augmentation
        neural_features_clone, perm, betas, select = mixco_1d(neural_features.clone())  
        neural_features_norm = nn.functional.normalize(neural_features_clone.flatten(1), dim=-1)
        img_features_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)
        
        # Clip loss calculation
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
        loss = loss_clip + regress_loss + prior_loss
        
        # Backward pass and optimization
        accelerator.backward(loss)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(unified_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(high_pipe.diffusion_prior.parameters(), 1.0)
        high_pipe.diffusion_prior.lr_scheduler.step()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        logits_img = logit_scale * neural_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        total += predicted.shape[0]
        correct += (predicted == labels).sum().item()
        
        # Clean up
        del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    # Calculate epoch metrics
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(epoch, unified_model, high_pipe, dataloader, device, 
                  text_features_all, img_features_all, k, config, eval_modality='eeg'):
    """
    Evaluation loop for one epoch
    
    Args:
        epoch: Current epoch number
        unified_model: Model to evaluate
        high_pipe: Diffusion prior pipeline
        dataloader: Data loader for evaluation
        device: Device to run on
        text_features_all: Precomputed text features
        img_features_all: Precomputed image features
        k: Number of classes to evaluate on
        config: Configuration dictionary
        eval_modality: Which modality to evaluate on
        
    Returns:
        Average loss, accuracy and top-5 accuracy
    """
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    
    # Select appropriate features based on modality
    if eval_modality=='eeg' or eval_modality=='fmri':
        img_features_all = (img_features_all[eval_modality]).to(device).float()
    elif eval_modality=='meg':
        img_features_all = (img_features_all[eval_modality][::12]).to(device).float()    
        
    # Initialize metrics
    total_loss = 0
    correct = 0
    total = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    loss_func = ClipLoss() 
    batch_idx = 0
    img_features_all = img_features_all.float()
    text_features_all = text_features_all.float()    
    num_voxels = {1: 6036, 2: 5944, 3: 5238} 
    mixup_pct = .33
    clip_scale = 1

    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, 
                       img_features, index, img_index, sub_ids) in enumerate(dataloader):
            
            # Move data to device
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Forward pass
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            logit_scale = unified_model.logit_scale.float()

            # Mixup augmentation
            neural_features, perm, betas, select = mixco_1d(neural_features)  
            neural_features_norm = nn.functional.normalize(neural_features.flatten(1), dim=-1)
            img_features_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)            
            
            # Clip loss calculation
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
            
            # Per-sample evaluation
            for idx, label in enumerate(labels):
                # Select k classes (k-1 incorrect + 1 correct)
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                
                if k == 200:
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    
                    # Top-5 accuracy calculation
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)                   
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count += 1                                
                    total += 1
                    
                elif k == 50 or k == 100:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                        
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)                   
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count += 1                                
                    total += 1
                    
                elif k == 2 or k == 4 or k == 10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
                    
            # Clean up
            del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids
    
    # Calculate final metrics
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(test_subjects, current_time, unified_model, high_pipe, 
                   train_dataloader, test_dataloader, optimizer, device, 
                   text_features_train_all, text_features_test_all, 
                   img_features_train_all, img_features_test_all, config, 
                   logger=None, accelerator=accelerator, eval_modality='eeg'):
    """
    Main training loop across all epochs
    
    Args:
        test_subjects: List of subject IDs for testing
        current_time: Timestamp for saving models
        unified_model: Main model to train
        high_pipe: Diffusion prior pipeline
        train_dataloader: Data loader for training
        test_dataloader: Data loader for evaluation
        optimizer: Optimizer
        device: Device to run on
        text_features_train_all: Precomputed text features for training
        text_features_test_all: Precomputed text features for testing
        img_features_train_all: Precomputed image features for training
        img_features_test_all: Precomputed image features for testing
        config: Configuration dictionary
        logger: WandB logger
        accelerator: Accelerator for distributed training
        eval_modality: Which modality to evaluate on
        
    Returns:
        List of results for each epoch
    """
    # Initialize logger if not provided
    logger = wandb_logger(config) if logger else None
    if logger:
        logger.watch(unified_model, logger) 
    
    # Initialize metrics tracking
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs, v4_accs, v10_accs = [], [], []
    
    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []
    scaler = GradScaler()
    
    # Setup diffusion prior if enabled
    if config.use_prior:
        high_pipe.diffusion_prior.train()
        high_pipe.diffusion_prior.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(train_dataloader) * config.epochs),
        )
    
    # Training loop
    for epoch in range(config.epochs):
        # Train for one epoch
        train_loss, train_accuracy = train_model(
            epoch, unified_model, high_pipe, train_dataloader, optimizer, device, 
            text_features_train_all, img_features_train_all, config=config, 
            accelerator=accelerator, eval_modality=eval_modality)
        
        # Periodic model saving
        if (epoch + 1) % 10 == 0:                    
            os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)             
            file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
            torch.save(unified_model.state_dict(), file_path)
            
            os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}/prior_diffusion", exist_ok=True)             
            prior_file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/prior_diffusion/{epoch+1}.pth"
            torch.save(high_pipe.diffusion_prior.state_dict(), prior_file_path)            
            print(f"Unified model saved in {file_path}!")
            print(f"prior diffusion model saved in {prior_file_path}!")
            
        # Track training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation
        if eval_modality == 'fmri':                
            test_loss, test_accuracy, top5_acc = evaluate_model(
                epoch, unified_model, high_pipe, test_dataloader, device, 
                text_features_test_all, img_features_test_all, k=100, 
                config=config, eval_modality=eval_modality)
        else:
            test_loss, test_accuracy, top5_acc = evaluate_model(
                epoch, unified_model, high_pipe, test_dataloader, device, 
                text_features_test_all, img_features_test_all, k=200, 
                config=config, eval_modality=eval_modality)    
            
        # Additional evaluations with different k values
        _, v2_acc, _ = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device, 
            text_features_test_all, img_features_test_all, k=2, 
            config=config, eval_modality=eval_modality)
        _, v4_acc, _ = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device, 
            text_features_test_all, img_features_test_all, k=4, 
            config=config, eval_modality=eval_modality)
        _, v10_acc, _ = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device, 
            text_features_test_all, img_features_test_all, k=10, 
            config=config, eval_modality=eval_modality)
        _, v50_acc, v50_top5_acc = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device, 
            text_features_test_all, img_features_test_all, k=50, 
            config=config, eval_modality=eval_modality)
        _, v100_acc, v100_top5_acc = evaluate_model(
            epoch, unified_model, high_pipe, test_dataloader, device, 
            text_features_test_all, img_features_test_all, k=100, 
            config=config, eval_modality=eval_modality)
            
        # Track evaluation metrics
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
        
        # Update best model info
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
            
        # Log to WandB
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

        # Print progress
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - "
              f"v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - "
              f"v100 Accuracy:{v100_acc}")

    # Plotting results
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    
    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # 2-class accuracy
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Best model info
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, 
                  ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    
    if logger:
        logger.finish()
    return results

def main():
    """Main function to setup and run training"""
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    
    # Model and data paths
    parser.add_argument(
        '--encoder_paths',
        nargs='+',
        required=False,
        default=[
            'eeg=./checkpoints/eeg_encoder.pth',
            'meg=./checkpoints/meg_encoder.pth',
            'fmri=./checkpoints/fmri_encoder.pth'
        ],
        help='Paths to pre-trained encoders (modify according to your checkpoint location)')           
    
    # Modality selection
    parser.add_argument('--modalities', nargs='+', choices=['eeg', 'meg', 'fmri'], 
                       default=['eeg', 'meg', 'fmri'], 
                       help='List of modalities to train on')
    parser.add_argument('--eval_modality', type=str, choices=['eeg', 'meg', 'fmri'], 
                       default='fmri', help='Modality to evaluate on')
    
    # Model architecture
    parser.add_argument('--depth', type=int, default=8, 
                       help='Depth of the model (default: 4)')
    
    # Dataset paths
    parser.add_argument('--eeg_data_path', type=str, 
                       default="./data/THINGS_EEG/Preprocessed_data_250Hz", 
                       help='Path to the EEG dataset (modify according to your dataset location)')
    parser.add_argument('--meg_data_path', type=str, 
                       default="./data/THINGS_MEG/preprocessed_newsplit", 
                       help='Path to the MEG dataset (modify according to your dataset location)')    
    parser.add_argument('--fmri_data_path', type=str, 
                       default="./data/fmri_dataset/Preprocessed", 
                       help='Path to the fMRI dataset (modify according to your dataset location)')      
    
    # Training configuration
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', 
                       help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", 
                       help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", 
                       help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", 
                       help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, 
                       help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:3', 
                       help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], 
                       default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, 
                       help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, 
                       default='Unified_EEG+MEG+fMRI_EEG', help='Encoder type')
    
    # Subject selection
    parser.add_argument('--test_subjects', nargs='+', default=['sub-02'], 
                       help='Subject ID to test on')        
    parser.add_argument('--eeg_subjects', nargs='+', 
                       default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
                               'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], 
                       help='List of EEG subject IDs')
    parser.add_argument('--meg_subjects', nargs='+', 
                       default=['sub-01', 'sub-02', 'sub-03', 'sub-04'], 
                       help='List of MEG subject IDs')    
    parser.add_argument('--fmri_subjects', nargs='+', 
                       default=['sub-01', 'sub-02', 'sub-03'], 
                       help='List of fMRI subject IDs')    
    parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, 
                       default=True, 
                       help="Whether to train diffusion prior")
    
    args = parser.parse_args()
    
    # Parse encoder paths
    encoder_paths = {}
    for path in args.encoder_paths:
        key, value = path.split('=')
        encoder_paths[key] = value
    
    # Initialize datasets based on selected modalities
    eeg_train_dataset = None
    meg_train_dataset = None
    fmri_train_dataset = None
    text_features_train_all = {}
    text_features_test_all = {}
    img_features_train_all = {}
    img_features_test_all = {}

    # Load datasets
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

    # Initialize models and training
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")    
    
    unified_model = UnifiedEncoder(
        encoder_paths, device, num_experts=5, 
        num_heads=args.depth, ff_dim=64*args.depth, 
        num_layers=args.depth)
    unified_model.to(device)

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    high_pipe = Pipe(diffusion_prior, device=device)
    high_pipe.diffusion_prior.to(device)
    
    # Print model info
    for name, param in unified_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")    
    
    total_params = sum(p.numel() for p in unified_model.parameters())
    trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
    print(f"Total parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")
    
    if total_params > 0:
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
    else:
        print("Total parameters count is zero, cannot compute percentage.")
            
    # Create data loaders
    metadataloader = MetaDataLoader(
        eeg_dataset=eeg_train_dataset if 'eeg' in args.modalities else None,
        meg_dataset=meg_train_dataset if 'meg' in args.modalities else None,
        fmri_dataset=fmri_train_dataset if 'fmri' in args.modalities else None,
        batch_size=args.batch_size,
        drop_last=True,
        modalities=args.modalities
    )
    train_loader = metadataloader

    # Create test dataset based on evaluation modality
    if args.eval_modality == 'eeg':
        test_dataset = EEGDataset(args.eeg_data_path, subjects=args.test_subjects, train=False)
    elif args.eval_modality == 'meg':
        test_dataset = MEGDataset(args.meg_data_path, subjects=args.test_subjects, train=False)
    elif args.eval_modality == 'fmri':
        test_dataset = fMRIDataset(args.fmri_data_path, subjects=args.test_subjects, train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, 
                            num_workers=0, drop_last=True)
    
    # Get test features
    text_features_test_all[args.eval_modality] = test_dataset.text_features
    img_features_test_all[args.eval_modality] = test_dataset.img_features    

    # Prepare for distributed training
    unified_model, diffusion_prior, optimizer, train_loader, test_loader = accelerator.prepare(
        unified_model, diffusion_prior, optimizer, train_loader, test_loader
    )
    
    # Run training
    results = main_train_loop(
        args.test_subjects, current_time, unified_model, high_pipe, 
        train_loader, test_loader, optimizer, device, 
        text_features_train_all, text_features_test_all, 
        img_features_train_all, img_features_test_all, 
        config=args, logger=args.logger, 
        accelerator=accelerator, eval_modality=args.eval_modality)

if __name__ == '__main__':
    main()