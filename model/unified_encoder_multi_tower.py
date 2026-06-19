'''
@File    :   unified_encoder_multi_tower.py
@Time    :   2025/07/13 18:40:53
@Author  :   DongyangLi
@Version :   1.0
@Desc    :   modified from [PAPER_NAME](https://arxiv.org/abs/XXXX.XXXXX) (CONFERENCE_ABBR'YY)
'''


import os
import sys
from dataclasses import dataclass
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from omegaconf import DictConfig, OmegaConf

# Import custom encoder modules
from .medformer_encoders import eeg_encoder, fmri_encoder, meg_encoder

# Load configuration file (relative to project root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
cfg = OmegaConf.load(os.path.join(_project_root, "configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Initialize linear layers with Xavier uniform initialization
default_linear_init = nn.init.xavier_uniform_


class HardMoEProjection(nn.Module):
    """
    Hard Mixture of Experts projection layer that selects only one expert per input.
    Args:
        input_dim: Input dimension size
        output_dim: Output dimension size
        num_experts: Number of expert networks
    """
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super(HardMoEProjection, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        # Single linear layer that computes outputs for all experts in parallel
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # Router network to select which expert to use
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts)
        )
        
    def forward(self, x):
        """
        Forward pass with hard expert selection.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Compute routing scores
        routing_scores = self.router(x)  # [batch_size, num_experts]
        
        # Select expert with highest score
        expert_indices = torch.argmax(routing_scores, dim=-1)  # [batch_size]
        
        # Create one-hot vector for selected expert
        routing_weights = torch.zeros_like(routing_scores)  # [batch_size, num_experts]
        routing_weights.scatter_(1, expert_indices.unsqueeze(-1), 1.0)
        
        # Compute all expert outputs in parallel
        experts_output = self.experts(x)  # [batch_size, num_experts * output_dim]
        
        # Reshape to [batch_size, num_experts, output_dim]
        experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
        # Apply routing weights
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        weighted_output = experts_output * routing_weights
        
        # Sum outputs (only selected expert contributes)
        output = weighted_output.sum(dim=1)  # [batch_size, output_dim]
        
        return output

    def get_selected_expert_indices(self, x):
        """
        Get indices of selected experts for analysis/debugging.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Tensor of expert indices [batch_size]
        """
        with torch.no_grad():
            routing_scores = self.router(x)
            expert_indices = torch.argmax(routing_scores, dim=-1)
        return expert_indices


class MoEProjection(nn.Module):
    """
    Soft Mixture of Experts projection layer that combines multiple experts.
    Args:
        input_dim: Input dimension size
        output_dim: Output dimension size
        num_experts: Number of expert networks
    """
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super(MoEProjection, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        # Single linear layer that computes outputs for all experts in parallel
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # Router network to compute weights for each expert
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
        )
        
    def forward(self, x):
        """
        Forward pass with soft expert combination.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Compute routing weights with sigmoid activation
        routing_weights = self.router(x).sigmoid()  # [batch_size, num_experts]
        
        # Compute all expert outputs in parallel
        experts_output = self.experts(x)  # [batch_size, num_experts * output_dim]
        experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Apply routing weights
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        weighted_output = experts_output * routing_weights
        
        # Sum weighted outputs
        output = weighted_output.sum(dim=1)  # [batch_size, output_dim]
        
        return output


class MoEProjection_upsamp(nn.Module):
    """
    Mixture of Experts projection layer with upsampling capability.
    Args:
        input_dim: Input dimension size
        output_dim: Output dimension size
        num_experts: Number of expert networks
    """
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super(MoEProjection_upsamp, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.clip_seq_dim = 256
        
        # Single linear layer that computes outputs for all experts in parallel
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Upsampling layer
        self.upsample = nn.Linear(1, self.clip_seq_dim)
        
        # Projection network
        self.projector = self._projector(output_dim * self.clip_seq_dim, output_dim)

    def _projector(self, in_dim, out_dim, h=512):
        """Helper function to create projection network."""
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )

    def forward(self, x):
        """
        Forward pass with upsampling.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            output: Projected output [batch_size, output_dim]
            upsampled_output: Upsampled output [batch_size, 257, output_dim]
        """
        # Compute routing weights
        routing_weights = self.router(x)  # [batch_size, num_experts]
        
        # Compute expert outputs
        experts_output = self.experts(x)  # [batch_size, num_experts * output_dim]
        experts_output = experts_output.view(x.size(0), self.num_experts, self.output_dim)
        
        # Apply routing weights
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        weighted_output = experts_output * routing_weights
        
        # Sum weighted outputs
        upsampled_output = weighted_output.sum(dim=1)  # [batch_size, output_dim]

        # Upsample output
        x_upsampled = upsampled_output.unsqueeze(2)  # [batch_size, output_dim, 1]
        upsampled_output = self.upsample(x_upsampled)  # [batch_size, output_dim, 257]
        upsampled_output = upsampled_output.permute(0, 2, 1)  # [batch_size, 257, output_dim]

        # Project upsampled output
        upsampled_output_flatten = upsampled_output.reshape(upsampled_output.size(0), -1)
        output = self.projector(upsampled_output_flatten)  # [batch_size, output_dim]

        return output, upsampled_output


class UnifiedEncoder(nn.Module):
    """
    Unified encoder that handles multiple modalities (EEG, MEG, fMRI) with MoE projection.
    Args:
        encoder_paths: Dictionary of paths to pretrained encoder weights
        device: Device to run on (default: cuda if available)
        in_dim: Input dimension size
        h: Hidden dimension size
        out_dim: Output dimension size
        num_experts: Number of experts in MoE
        num_heads: Number of attention heads
        ff_dim: Feedforward dimension
        num_layers: Number of transformer layers
        user_caption: Whether to use captioning (affects output)
    """
    def __init__(self, encoder_paths: dict[str, str] = None, device=None, 
                 in_dim=1024, h=1024, out_dim=1024, num_experts=5, 
                 num_heads=4, ff_dim=2048, num_layers=4, user_caption=False):
        super().__init__()
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.user_caption = user_caption
        
        # Initialize MoE projection based on captioning requirement
        if self.user_caption:
            self.moe_projection = MoEProjection_upsamp(
                input_dim=in_dim, 
                output_dim=out_dim, 
                num_experts=num_experts
            )
        else: 
            self.moe_projection = MoEProjection(
                input_dim=in_dim, 
                output_dim=out_dim, 
                num_experts=num_experts
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))          
        self.modals = ['eeg', 'meg', 'fmri']
                    
        self.encoder = nn.ModuleDict()
        
        # Initialize or load pretrained encoders for each modality
        for modal in self.modals:
            if modal == 'eeg':
                encoder = eeg_encoder()
                if encoder_paths is not None and 'eeg' in encoder_paths:
                    encoder.load_state_dict(torch.load(encoder_paths['eeg'], map_location=self.device))
                    encoder.eval()  # Set to eval mode for pretrained weights
                    for param in encoder.parameters():  # Freeze parameters
                        param.requires_grad = False
                encoder.to(self.device)
                self.encoder[modal] = encoder
                
            elif modal == 'meg':
                encoder = meg_encoder()
                if encoder_paths is not None and 'meg' in encoder_paths:
                    encoder.load_state_dict(torch.load(encoder_paths['meg'], map_location=self.device))
                    encoder.eval()
                    for param in encoder.parameters():
                        param.requires_grad = False
                encoder.to(self.device)
                self.encoder[modal] = encoder
    
            elif modal == 'fmri':                
                encoder = fmri_encoder()
                if encoder_paths is not None and 'fmri' in encoder_paths:
                    encoder.load_state_dict(torch.load(encoder_paths['fmri'], map_location=self.device))
                    encoder.eval()
                    for param in encoder.parameters():
                        param.requires_grad = False
                encoder.to(self.device)
                self.encoder[modal] = encoder

    def forward(self, x, subject_ids, modal):        
        """Forward pass through the appropriate encoder and projection."""
        x = self.encoder[modal](x, subject_ids)                                

        if self.user_caption:
            output, upsampled_output = self.moe_projection(x)
            return output, upsampled_output
        else:
            output = self.moe_projection(x)
            return output
