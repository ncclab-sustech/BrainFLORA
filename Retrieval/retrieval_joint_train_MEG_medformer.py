import os
import re
import csv
import math
import random
import argparse
import datetime
from itertools import combinations
from typing import Tensor

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import CrossEntropyLoss, functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
import tqdm

# Set environment variables for Weights & Biases
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

# Import custom modules
from megdatasets_joint_subjects import MEGDataset
from utils import wandb_logger
from utils.losses import ClipLoss
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet

# Import custom transformer components
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from layers.Medformer_EncDec import Encoder as MedformerEncoder
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding


class Config:
    """Configuration class for model parameters"""
    def __init__(self):
        self.task_name = 'classification'  # Task type
        self.seq_len = 201                 # Input sequence length
        self.pred_len = 250               # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                # Model dimension
        self.embed = 'timeF'              # Time encoding method
        self.freq = 'h'                   # Time frequency
        self.dropout = 0.25               # Dropout rate
        self.factor = 1                   # Attention scaling factor
        self.n_heads = 4                  # Number of attention heads
        self.e_layers = 1                 # Number of encoder layers
        self.d_ff = 256                   # Feedforward dimension
        self.activation = 'gelu'          # Activation function
        self.enc_in = 271                 # Encoder input dimension
        self.single_channel = False       # Single channel mode
        self.patch_len_list = "2,4,8"     # Patch lengths for embedding
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"  # Data augmentations
        self.no_inter_attn = False        # Disable inter-attention
        self.num_class = 250              # Number of classes


class Medformer(nn.Module):
    """
    Medformer model with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
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
        
        # Transformer encoder
        self.encoder = MedformerEncoder(
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
                configs.d_model * sum(patch_num_list) * 
                (1 if not self.single_channel else configs.enc_in),
                configs.num_class,
            )

    def classification(self, x_enc, x_mark_enc):
        """Forward pass for classification task"""
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))
            
        output = self.act(enc_out)
        output = self.dropout(output)
        return output

    def forward(self, x_enc, x_mark_enc=None):
        """Main forward pass"""
        return self.classification(x_enc, x_mark_enc)


class iTransformer(nn.Module):
    """iTransformer model implementation"""
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # Embedding layer
        self.enc_embedding = DataEmbedding(
            configs.seq_len, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout,
            joint_train=joint_train,
            num_subjects=num_subjects
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, 
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) 
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        """Forward pass"""
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]  # Truncate output
        return enc_out


class PatchEmbedding(nn.Module):
    """EEG signal patch embedding module"""
    def __init__(self, emb_size=40):
        super().__init__()
        # Temporal-spatial convolution from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (178, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        
        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.tsconv(x)
        return self.projection(x)


class ResidualAdd(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass with residual connection"""
        res = x
        x = self.fn(x, **kwargs)
        return x + res


class FlattenHead(nn.Sequential):
    """Flattening layer for classification head"""
    def forward(self, x):
        """Flatten input tensor"""
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
    """ATMS (Adaptive Transformer for Multi-Subject) model"""
    def __init__(self, sequence_length=250, num_subjects=10, joint_train=False):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = Medformer(default_config)
        self.subject_wise_linear = nn.ModuleList([
            nn.Linear(default_config.d_model, sequence_length) 
            for _ in range(num_subjects)
        ])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        """Forward pass"""
        x = self.encoder(x)
        eeg_embedding = self.enc_eeg(x)
        return self.proj_eeg(eeg_embedding)


def extract_id_from_string(s):
    """Extract numeric ID from subject string (e.g., 'sub-01' -> 1)"""
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None


def train_model(eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config):
    """Training loop for one epoch"""
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::12]).to(device).float()
    
    total_loss, correct, total = 0, 0, 0
    features_list = []  # Store features if needed
    
    for batch_idx, (_, eeg_data, labels, text, text_features, img, img_features, sub_ids) in enumerate(dataloader):
        # Move data to device
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        # Prepare subject IDs
        subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
        subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        eeg_features = eeg_model(eeg_data, subject_ids).float()
        features_list.append(eeg_features)
        
        # Compute loss
        logit_scale = eeg_model.logit_scale
        eeg_features_norm = F.normalize(eeg_features.flatten(1), dim=-1)
        img_features_norm = F.normalize(img_features.flatten(1), dim=-1)
        img_loss = eeg_model.loss_func(eeg_features_norm, img_features_norm, logit_scale)
        
        # Backward pass
        loss = img_loss
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        logits_img = logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        
        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    return total_loss / (batch_idx+1), correct / total, torch.cat(features_list, dim=0)


def evaluate_model(eeg_model, dataloader, device, text_features_all, img_features_all, k, config):
    """Evaluation loop"""
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    
    total_loss, correct, total = 0, 0, 0
    top5_correct_count = 0
    
    with torch.no_grad():
        for batch_idx, (_, eeg_data, labels, text, text_features, img, img_features, sub_ids) in enumerate(dataloader):
            # Move data to device
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            # Prepare subject IDs
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Forward pass
            eeg_features = eeg_model(eeg_data, subject_ids).float()
            
            # Compute loss
            logit_scale = eeg_model.logit_scale
            eeg_features_norm = F.normalize(eeg_features.flatten(1), dim=-1)
            img_features_norm = F.normalize(img_features.flatten(1), dim=-1)
            img_loss = eeg_model.loss_func(eeg_features_norm, img_features_norm, logit_scale)
            total_loss += img_loss.item()
            
            # Evaluate for each sample in batch
            for idx, label in enumerate(labels):
                possible_classes = list(set(range(text_features_all.size(0))) - {label.item()}
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                # Compute logits
                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                predicted_label = selected_classes[torch.argmax(logits_img).item()]
                
                # Update metrics
                if predicted_label == label.item():
                    correct += 1
                
                # Top-5 accuracy
                _, top5_indices = torch.topk(logits_img, 5, largest=True)
                if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                    top5_correct_count += 1
                
                total += 1

    return total_loss / (batch_idx+1), correct / total, top5_correct_count / total


def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    """Main training loop with logging and evaluation"""
    if logger:
        logger = wandb_logger(config)
        logger.watch(eeg_model, logger)

    # Initialize metrics tracking
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs, v4_accs, v10_accs = [], [], []
    best_accuracy, best_epoch_info = 0.0, {}
    results = []

    for epoch in range(config.epochs):
        # Training phase
        train_loss, train_accuracy, features_tensor = train_model(
            eeg_model, train_dataloader, optimizer, device, 
            text_features_train_all, img_features_train_all, config
        )
        
        # Periodic model saving
        if (epoch + 1) % 5 == 0:
            save_dir = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}" if config.insubject else f"./models/contrast/across/{config.encoder_type}/{current_time}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(eeg_model.state_dict(), f"{save_dir}/{epoch+1}.pth")
        
        # Evaluation phase
        test_loss, test_accuracy, top5_acc = evaluate_model(
            eeg_model, test_dataloader, device, 
            text_features_test_all, img_features_test_all, 200, config
        )
        _, v2_acc, _ = evaluate_model(eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, 2, config)
        _, v4_acc, _ = evaluate_model(eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, 4, config)
        _, v10_acc, _ = evaluate_model(eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, 10, config)
        _, v50_acc, v50_top5_acc = evaluate_model(eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, 50, config)
        _, v100_acc, v100_top5_acc = evaluate_model(eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, 100, config)
        
        # Update metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Log results
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
        
        # Update best model
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
        
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Top5 Accuracy: {top5_acc:.4f}")
        print(f"v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - "
              f"v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")

    # Plot results
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Add best model info
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
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    
    if logger:
        logger.finish()
    
    return results


def main():
    """Main function to parse arguments and run training"""
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--data_path', type=str, default='./data/THINGS_MEG/preprocessed_npy', help='Path to data (modify according to your dataset location)')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Output directory')
    parser.add_argument('--project', type=str, default='train_pos_img_text_rep', help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size')
    parser.add_argument('--insubject', default=False, help='Within-subject training flag')
    parser.add_argument('--encoder_type', type=str, default='ATMS', 
                        choices=['ATMS', 'EEGNetv4_Encoder', 'ATCNet_Encoder', 'EEGConformer_Encoder', 'EEGITNet_Encoder', 'ShallowFBCSPNet_Encoder'], 
                        help='Encoder type')
    parser.add_argument('--logger', default=True, help='Enable logging')
    parser.add_argument('--gpu', type=str, default='cuda:', help='GPU device')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device type')
    parser.add_argument('--joint_train', action='store_true', help='Joint subject training flag')
    parser.add_argument('--sub', type=str, default='sub-01', help='Subject ID')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04'], help='Subject IDs')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.gpu if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    
    # Initialize model and data
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    eeg_model = globals()[args.encoder_type](joint_train=True)
    eeg_model.to(device)
    optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=args.lr)
    
    # Load datasets
    train_dataset = MEGDataset(args.data_path, adap_subject=args.sub, subjects=args.subjects, train=True)
    test_dataset = MEGDataset(args.data_path, adap_subject=args.sub, subjects=args.subjects, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # Get features
    text_features_train_all = train_dataset.text_features
    text_features_test_all = test_dataset.text_features
    img_features_train_all = train_dataset.img_features
    img_features_test_all = test_dataset.img_features

    # Run training
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