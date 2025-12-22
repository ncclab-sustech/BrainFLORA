import os
import re
import csv
import math
import random
import argparse
import datetime
from itertools import combinations
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch import Tensor
# CLIP and custom imports
import clip
import torchvision.transforms as transforms
from fmri_datasets_joint_subjects import fMRIDataset
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
from utils.losses import ClipLoss
from utils import wandb_logger
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding

# Set up environment variables for Weights & Biases
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = "offline"

class Config:
    """Configuration class for model parameters"""
    def __init__(self, depth=1):
        self.task_name = 'classification'
        self.seq_len = 1024
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1 * depth / 5
        self.factor = 1
        self.n_heads = depth
        self.e_layers = depth
        self.d_ff = 64 * depth
        self.activation = 'gelu'
        self.enc_in = 8
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250

class Medformer(nn.Module):
    """
    Medformer model architecture for medical time series classification
    Based on Transformer with O(L^2) complexity
    Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, configs):
        super(Medformer, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel
        
        # Patch embedding configuration
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")
        
        # Embedding layer
        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )
        
        # Encoder layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        
        # Classification head
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * sum(patch_num_list) * (1 if not self.single_channel else configs.enc_in),
                configs.num_class,
            )

    def forward(self, x_enc, x_mark_enc=None):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """Classification task forward pass"""
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        output = self.act(enc_out)
        output = self.dropout(output)
        return output

class PatchEmbedding(nn.Module):
    """Patch embedding module for EEG data"""
    def __init__(self, emb_size=40):
        super().__init__()
        # Temporal-spatial convolution from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (899, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)     
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    """Flatten layer for sequence to vector conversion"""
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class Enc_eeg(nn.Sequential):
    """EEG encoder module"""
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    """Projection head for EEG features"""
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS(nn.Module):
    """ATMS model combining Medformer and EEG processing"""
    def __init__(self, sequence_length=250, num_subjects=10, joint_train=False, configs=None):
        super(ATMS, self).__init__()
        default_config = Config(configs.depth)
        self.encoder = Medformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([
            nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)
        ])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
        self.fmri_subs = [i for i in range(1, 4)]
        self.num_voxels = {1: 10999, 2: 10212, 3: 8128}             
        self.conv1_fmri = nn.ModuleDict({
            str(sub): nn.Linear(11000, 8192) for sub in self.fmri_subs
        })    
    
    def forward(self, x, subject_ids):
        x = self.conv1_fmri[f'{subject_ids[0].item()}'](x)
        x = x.reshape(x.size(0), -1, 1024)        
        x = self.encoder(x)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out  

def extract_id_from_string(s):
    """Extract subject ID from string"""
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None

def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config):
    """Training loop for one epoch"""
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::12]).to(device).float()
    
    total_loss, correct, total = 0, 0, 0
    alpha = 0.99
    features_list = []
    
    for batch_idx, (_, eeg_data, labels, text, text_features, img, img_features, _) in enumerate(dataloader):
        # Move data to device
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        batch_size = eeg_data.size(0)
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)        
        eeg_features = eeg_model(eeg_data, subject_ids).float()
        features_list.append(eeg_features)
        
        # Compute losses
        logit_scale = eeg_model.logit_scale
        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics calculation
        total_loss += loss.item()
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)
        
        total += predicted.shape[0]
        correct += (predicted == labels).sum().item()
        
        # Clean up
        del eeg_data, labels, text, text_features, img, img_features
        
    return total_loss / (batch_idx+1), correct / total, torch.cat(features_list, dim=0)

def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k, config):
    """Evaluation loop"""
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    alpha = 0.99
    total_loss, correct, total = 0, 0, 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    
    with torch.no_grad():
        for batch_idx, (_, eeg_data, labels, text, text_features, img, img_features, _) in enumerate(dataloader):
            # Move data to device
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            # Forward pass
            batch_size = eeg_data.size(0)
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)        
            eeg_features = eeg_model(eeg_data, subject_ids).float()
            
            # Compute losses
            logit_scale = eeg_model.logit_scale 
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            loss = img_loss * alpha + text_loss * (1-alpha)
            total_loss += loss.item()
            
            # Evaluate for each sample in batch
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                if k == 100:
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    
                    # Top-5 accuracy calculation
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count += 1                                 
                    total += 1
                
                elif k in [2, 4, 10, 50, 100]:
                    # Similar evaluation for other k values
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    
                    if k in [50, 100]:
                        _, top5_indices = torch.topk(logits_single, 5, largest=True)
                        if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                            top5_correct_count += 1                                 
                    total += 1
                
                else:
                    print("Error in k value")
            
            # Clean up
            del eeg_data, labels, text, text_features, img, img_features        
    
    return total_loss / (batch_idx+1), correct / total, top5_correct_count / total

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    """Main training loop across epochs"""
    if logger:
        logger = wandb_logger(config)
        logger.watch(eeg_model, logger)

    # Initialize tracking variables
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs, v4_accs, v10_accs = [], [], []
    best_accuracy, best_model_weights = 0.0, None
    best_epoch_info = {}
    results = []

    for epoch in range(config.epochs):
        # Training phase
        train_loss, train_accuracy, features_tensor = train_model(
            sub, eeg_model, train_dataloader, optimizer, device, 
            text_features_train_all, img_features_train_all, config
        )
        
        # Save model periodically
        if (epoch + 1) % 5 == 0:                    
            save_dir = f"./models/contrast/{'across' if not config.insubject else sub}/{config.encoder_type}/{current_time}"
            os.makedirs(save_dir, exist_ok=True)             
            torch.save(eeg_model.state_dict(), f"{save_dir}/{epoch+1}.pth")
            print(f"Model saved in {save_dir}!")

        # Evaluation phase
        test_loss, test_accuracy, top5_acc = evaluate_model(
            sub, eeg_model, test_dataloader, device, 
            text_features_test_all, img_features_test_all, k=100, config=config
        )
        
        # Additional evaluations for different k values
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=2, config=config)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=4, config=config)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=10, config=config)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=50, config=config)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=100, config=config)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Log epoch results
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

        # Log to wandb
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

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
    # Plot results
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    plot_metrics(axs, train_losses, test_losses, train_accuracies, test_accuracies, 
                v2_accs, v4_accs, v10_accs, best_epoch_info)
    
    plt.tight_layout()
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    
    return results

def plot_metrics(axs, train_losses, test_losses, train_accuracies, test_accuracies, 
                v2_accs, v4_accs, v10_accs, best_epoch_info):
    """Helper function to plot training metrics"""
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

    # k-class accuracy curves
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Best epoch info
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

def main():
    """Main function to parse arguments and run training"""
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--data_path', type=str, default='./data/fmri_dataset/Preprocessed', help='Path to data (modify according to your dataset location)')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')
    parser.add_argument('--project', type=str, default='train_pos_img_text_rep', help='Project name for logging')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=150, help='Batch size for training')
    parser.add_argument('--insubject', default=False, help='Flag to indicate within-subject training')
    parser.add_argument('--encoder_type', type=str, default='ATMS', 
                       choices=['ATMS', 'EEGNetv4_Encoder', 'ATCNet_Encoder', 'EEGConformer_Encoder', 'EEGITNet_Encoder', 'ShallowFBCSPNet_Encoder'], 
                       help='Encoder type')
    parser.add_argument('--logger', default=True, help='Enable logging')
    parser.add_argument('--gpu', type=str, default='cuda:2', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on')
    parser.add_argument('--joint_train', action='store_true', help='Joint subject training flag')
    parser.add_argument('--sub', type=str, default='sub-01', help='Subject ID for testing')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03'], help='List of subject IDs')
    parser.add_argument('--depth', type=int, default=4, help='Number of encoder layers')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.gpu if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    
    # Initialize model and data
    eeg_model = globals()[args.encoder_type](joint_train=True, configs=args)
    eeg_model.to(device)
    optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=args.lr)
    
    # Prepare datasets
    train_dataset = fMRIDataset(args.data_path, adap_subject=args.sub, subjects=args.subjects, train=True)
    test_dataset = fMRIDataset(args.data_path, adap_subject=args.sub, subjects=args.subjects, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # Get features
    text_features_train_all = train_dataset.text_features
    text_features_test_all = test_dataset.text_features
    img_features_train_all = train_dataset.img_features
    img_features_test_all = test_dataset.img_features

    # Run training
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    results = main_train_loop(
        args.sub, current_time, eeg_model, train_loader, test_loader, optimizer, device, 
        text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, 
        config=args, logger=args.logger
    )

    # Save results
    results_dir = os.path.join(args.output_dir, args.encoder_type, args.sub, current_time)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{args.encoder_type}_ada_exclude_{args.sub}.csv")

    with open(results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()