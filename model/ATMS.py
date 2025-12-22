import os
import argparse
import datetime
import time
from pathlib import Path
import functools
import multiprocessing
import random
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import tqdm
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix

# Import custom modules
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from utils.losses import ClipLoss
import utils.misc as misc

# Set environment variables
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
cudnn.benchmark = True


class Config:
    """Configuration class for model hyperparameters."""
    def __init__(self):
        self.task_name = 'classification'  # Task type (classification/regression/etc)
        self.seq_len = 1024                # Input sequence length
        self.pred_len = 1024               # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 1024                # Dimension of model embeddings
        self.embed = 'timeF'              # Type of embedding (time features)
        self.freq = 'h'                   # Frequency for time features
        self.dropout = 0.25               # Dropout rate
        self.factor = 1                   # Attention factor
        self.n_heads = 4                  # Number of attention heads
        self.e_layers = 1                 # Number of encoder layers
        self.d_ff = 256                  # Dimension of feed-forward network
        self.activation = 'gelu'          # Activation function


class iTransformer(nn.Module):
    """iTransformer model architecture for time series data."""
    def __init__(self, configs, joint_train=False, num_subjects=None):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # Embedding layer
        self.enc_embedding = DataEmbedding(
            configs.seq_len, configs.d_model, configs.embed, 
            configs.freq, configs.dropout, 
            joint_train=joint_train, num_subjects=num_subjects
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, 
                            attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
    
    def forward(self, x_enc, x_mark_enc, subject_ids=None, modal='eeg'):
        """
        Forward pass for the iTransformer.
        
        Args:
            x_enc: Input tensor
            x_mark_enc: Time feature markers
            subject_ids: Subject IDs for personalized embeddings
            modal: Modality type ('eeg', 'meg', or 'fmri')
            
        Returns:
            enc_out: Encoded output
            attns: Attention weights (if output_attention=True)
        """
        enc_out, attns = self.encoder(x_enc, attn_mask=None)
        
        # Select relevant channels based on modality
        if modal == 'eeg':
            enc_out = enc_out[:, :54, :]      
        elif modal == 'meg':
            enc_out = enc_out[:, :262, :]        
        elif modal == 'fmri':
            enc_out = enc_out[:, :8, :]                      
            
        return enc_out


class PatchEmbedding(nn.Module):
    """Patch embedding module inspired by ShallowNet architecture."""
    def __init__(self, emb_size=40):
        super().__init__()
        # Temporal-spatial convolution block
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
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
        """Forward pass through patch embedding."""
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    """Residual connection wrapper."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass with residual connection."""
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    """Flatten layer for reshaping tensors."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Flatten input tensor."""
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    """EEG encoder module combining patch embedding and flattening."""
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    """Projection head for EEG features."""
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
    """Attention-based Temporal Modeling System (ATMS) for multimodal data."""
    def __init__(self, sequence_length=1024, num_features=64, num_latents=1024, 
                 num_blocks=1, joint_train=False, num_subjects=10):
        super(ATMS, self).__init__()
        # Initialize components
        default_config = Config()
        self.encoder = iTransformer(default_config, joint_train=joint_train, num_subjects=num_subjects)
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        
        # CLIP-style loss parameters
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    
    def forward(self, x, subject_ids, modal):
        """
        Forward pass for ATMS.
        
        Args:
            x: Input tensor
            subject_ids: Subject identifiers
            modal: Data modality ('eeg', 'meg', 'fmri')
            
        Returns:
            Encoded features
        """
        x = self.encoder(x, None, subject_ids, modal)
        return x