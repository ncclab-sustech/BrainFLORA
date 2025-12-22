import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from utils.losses import ClipLoss

# Import from installed package (use `pip install -e .` from project root)
from model.Medformer import Medformer
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding

class Config:
    def __init__(self):
        self.task_name = 'classification'      
        self.seq_len = 201                     
        self.pred_len = 250                     
        self.output_attention = False          
        self.d_model = 250                     
        self.embed = 'timeF'                   
        self.freq = 'h'                        
        self.dropout = 0.25                    
        self.factor = 1                        
        self.n_heads = 4                       
        self.e_layers = 1                     
        self.d_ff = 256                       
        self.activation = 'gelu'               
        self.enc_in = 271                      
        
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Remove temporal-spatial convolution, use linear layers instead
        self.linear_layers = nn.Sequential(
            nn.Linear(250, 128),  # First reduce time dimension
            nn.BatchNorm1d(178),  # Batch normalization on channel dimension
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 40),   # Reduce to final required dimension
            nn.BatchNorm1d(178),
            nn.ELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        # print(f"PatchEmbedding input shape: {x.shape}")  # [32, 178, 250]
        
        # Adjust dimension order to fit linear layers
        x = x.transpose(1, 2)  # [32, 250, 178]
        x = x.transpose(1, 2)  # [32, 178, 250]
        
        x = self.linear_layers(x)  # [32, 178, 40]
        # print(f"After linear layers shape: {x.shape}")
        
        return x  # [32, 178, 40]

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # print(f"FlattenHead input shape: {x.shape}")
        x = x.contiguous().view(x.size(0), -1)  # [32, 178*40]
        # print(f"FlattenHead output shape: {x.shape}")
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=7120, proj_dim=1024, drop_proj=0.5):  # Modified embedding_dim=178*40=7120
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class meg_encoder(nn.Module):    
    def __init__(self, sequence_length=250, num_subjects=10, joint_train=False):
        super(meg_encoder, self).__init__()
        default_config = Config()
        self.encoder = Medformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        # print(f"Input shape: {x.shape}")  # [32, 271, 201]
        
        x = self.encoder(x)  # [32, 178, 250]
        # print(f"After encoder shape: {x.shape}")
        
        eeg_embedding = self.enc_eeg(x)  # [32, 7120]
        # print(f"After enc_eeg shape: {eeg_embedding.shape}")
        
        out = self.proj_eeg(eeg_embedding)  # [32, 1024]
        # print(f"Final output shape: {out.shape}")
        
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# Test code
if __name__ == "__main__":
    # Create a random input tensor
    batch_size = 32
    channels = 271
    time_steps = 201
    x = torch.randn(batch_size, channels, time_steps)
    subject_ids = torch.zeros(batch_size)

    # Initialize model
    model = meg_encoder()
    
    # Forward pass
    output = model(x, subject_ids)
    
    print("\nFinal output size:", output.shape)