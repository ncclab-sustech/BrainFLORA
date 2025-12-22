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

class Config:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 250
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
        
        self.enc_in = 63
        
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=1024):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)  # B*55250 after attn and flatten
        self.project = nn.Sequential(
            nn.Linear(55250, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(0.5),
            
        )

    def forward(self, x):
        # x: [b,63,250]
        x = self.flatten(x)      # [b,15750]
        x = self.project(x)      # [b,emb_size]
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
    def __init__(self, emb_size=1440, **kwargs):
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

class eeg_encoder(nn.Module):    
    def __init__(self, sequence_length=250, num_subjects=10, joint_train=False):
        super(eeg_encoder, self).__init__()
        default_config = Config()
        self.encoder = Medformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
 
    def forward(self, x, subject_ids=None):        
        x = self.encoder(x)        
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out
    
