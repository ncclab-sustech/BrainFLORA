import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from loss import ClipLoss

# Import from installed package (use `pip install -e .` from project root)
from model.Medformer import Medformer
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding
    
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
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
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
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


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name        
        self.seq_len = 1024                      # Sequence length
        self.pred_len = 250                     # Prediction length        
        self.output_attention = False          # Whether to output attention weights
        self.d_model = 250                     # Model dimension
        self.embed = 'timeF'                   # Time encoding method
        self.freq = 'h'                        # Time frequency
        self.dropout = 0.25                    # Dropout ratio
        self.factor = 1                        # Attention scaling factor
        self.n_heads = 4                       # Number of attention heads
        self.e_layers = 1                     # Number of encoder layers
        self.d_ff = 256                       # Feedforward network dimension
        self.activation = 'gelu'               # Activation function
        self.enc_in = 8                         # Encoder input dimension (example value)
        
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250


class fmri_encoder(nn.Module):    
    def __init__(self, sequence_length=250, num_subjects=10, joint_train=False):
        super(fmri_encoder, self).__init__()
        default_config = Config()
        self.encoder = Medformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
        self.fmri_subs = [i for i in range(1, 4)]
        self.num_voxels = {1: 6036, 2: 5944, 3: 5238}                
        # self.conv1_fmri = nn.ModuleDict({
        #             str(sub): nn.Linear(7000, 8192) for sub in self.fmri_subs})    
        self.proj_fmri = nn.Linear(7000, 8192)
    def forward(self, x, subject_ids):
        # x = self.conv1_fmri[f'{subject_ids[0].item()}'](x)
        x = self.proj_fmri(x)           
        x = x.reshape(x.size(0), -1, 1024)        
        x = self.encoder(x)
        eeg_embedding = self.enc_eeg(x)        
        out = self.proj_eeg(eeg_embedding)
        return out  
    
if __name__ == "__main__":
    # Create a random input tensor
    batch_size = 32
    x = torch.randn(batch_size, 7000)
    subject_ids = torch.zeros(batch_size)

    # Initialize model
    model = fmri_encoder()
    
    # Forward pass
    output = model(x, subject_ids)
    print(output.shape)