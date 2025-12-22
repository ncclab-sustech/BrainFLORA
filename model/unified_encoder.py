import os
import math
import sys
import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from einops.layers.torch import Rearrange
from omegaconf import OmegaConf
import hydra
import open_clip
from flash_attn import flash_attn_func
from transformers import CLIPVisionModel

# Import custom modules
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from loss import ClipLoss
from .components import RMSNorm
from .perceiver import PerceiverResampler
from .ATMS import ATMS
from .Medformer import Medformer
from .projector import FusionHead

# Load configuration (relative to project root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
cfg = OmegaConf.load(os.path.join(_project_root, "configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Initialize linear layers with Xavier uniform initialization
default_linear_init = nn.init.xavier_uniform_

@dataclass
class ModelArgs:
    """Configuration class for model hyperparameters"""
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    token_size: int = 1024  # defined later by tokenizer
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequency cis values for rotary embeddings"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency cis for broadcasting"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    """Multi-head attention module with flash attention support"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # Linear projections for queries, keys, values and output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        default_linear_init(self.wq.weight)
        
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        default_linear_init(self.wk.weight)
        
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        default_linear_init(self.wv.weight)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        default_linear_init(self.wo.weight)

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor], prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Apply rotary embeddings
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Handle key-value cache
        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        # Flash attention
        output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        """Allocate key-value cache for inference"""
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        """Clear key-value cache"""
        self.k_cache, self.v_cache = None, None

class FeedForward(nn.Module):
    """Feed-forward network with gated activation"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        default_linear_init(self.w1.weight)
        
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        default_linear_init(self.w2.weight)
        
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        default_linear_init(self.w3.weight)

    def _silu_gating(self, x, y):
        """SILU gating mechanism"""
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers"""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        """Forward pass for feed-forward network"""
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        """Forward pass for attention layer"""
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor], prompt=None):
        """Complete forward pass for transformer block"""
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)
        return out

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        default_linear_init(self.fc1.weight)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        default_linear_init(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class BrainEncoder(nn.Module):
    """Encoder for brain data using CLIP vision model"""
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')        
        self.clip_size = (224, 224)       
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc

        # Freeze CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        self.clip_width = self.clip.vision_model.embeddings.patch_embedding.out_channels

        self.conv1 = nn.ModuleDict()
        self.position_embedding = nn.ParameterDict()
        self.modals = ['image', 'fmri']
        for modal in self.modals:
            if modal =='image':
                modal_tokens = 256 + 1
            elif modal == 'fmri':
                modal_tokens = 8 + 1
                self.conv1[modal] = nn.Linear(15724, 8192)
                self.position_embedding[modal] = nn.Embedding(modal_tokens, self.clip_width)

    def clip_encode_image(self, x, modal='image'):
        """Encode image using CLIP model"""
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) 

        # Add class embedding
        x = torch.cat([self.clip.vision_model.embeddings.class_embedding.to(x.dtype) + 
                      torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  

        pos_embedding = self.clip.vision_model.embeddings.position_embedding
        if modal == 'fmri':
            pos_embedding = self.position_embedding[modal]
            
        modal_tokens = 257 if modal == 'image' else 9
        position_ids = torch.arange(0, modal_tokens).unsqueeze(0).to(x.device)

        x = x + pos_embedding(position_ids)
        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)

        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]
        image_features = select_hidden_state[:, 1:]  # Remove class token

        return image_features

    def encode_image(self, x, modal='image'):
        """Encode image or fMRI data"""
        if modal in ['image']:
            x = self.preprocess(x)
            x = self.clip.vision_model.embeddings.patch_embedding(x)
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            x = x.reshape(x.size(0), self.clip_width, -1)

        image_feats = self.clip_encode_image(x, modal=modal)
        return image_feats

class Perceiver(nn.Module):
    """Perceiver resampler for processing visual features"""
    def __init__(self, patch_embed_dim=1024, hidden_size=512, num_latents=1024):
        super().__init__()
        self.ln_vision = nn.LayerNorm(patch_embed_dim)
        self.llm_proj = nn.Linear(patch_embed_dim, hidden_size)
        self.perceiver = PerceiverResampler(
            dim=patch_embed_dim,
            dim_head=96,
            depth=6,
            heads=16,
            num_latents=num_latents,
            num_media_embeds=1
        )

    def forward(self, image_features):
        """Forward pass for perceiver"""
        image_features = self.ln_vision(image_features)
        inputs_llm = self.perceiver(image_features)
        return self.llm_proj(inputs_llm)

class MoEProjection(nn.Module):
    """Mixture of Experts projection layer"""
    def __init__(self, input_dim=250, output_dim=1024, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """Forward pass with expert routing"""
        batch_size, num_latents, _ = x.shape
        routing_weights = self.router(x)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            weighted_output = expert_output * routing_weights[:, :, i].unsqueeze(-1)
            expert_outputs.append(weighted_output)

        output = torch.stack(expert_outputs, dim=0).sum(dim=0)
        return output

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
    """Flatten layer for sequence processing"""
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class Enc_eeg(nn.Sequential):
    """EEG encoder with patch embedding"""
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class PatchEmbedding(nn.Module):
    """Patch embedding layer for EEG data"""
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (221, 1), stride=(1, 1)),
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

class Proj_neuro(nn.Sequential):
    """Projection layer for neural data"""
    def __init__(self, emb_size=40, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead(),
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Proj_eeg(nn.Sequential):
    """Projection layer for EEG data"""
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
    """Configuration class for model parameters"""
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
        self.enc_in = 256
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250

class MoEMedformer(nn.Module):
    """Mixture of Experts Medformer model"""
    def __init__(self, input_dim=250, output_dim=250, num_experts=3, 
                 medformer_config=None, modalities=['eeg', 'meg', 'fmri']):
        super(MoEMedformer, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.modalities = modalities
        
        if medformer_config is None:
            raise ValueError("medformer_config must be provided")
            
        self.experts = nn.ModuleList([Medformer(medformer_config) for _ in range(num_experts)])
        self.lambda_params = nn.ParameterDict({
            modality: nn.Parameter(torch.ones(num_experts, output_dim)) for modality in modalities
        })

        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, modality):
        """Forward pass with modality-specific expert weighting"""
        if modality not in self.modalities:
            raise ValueError(f"Unsupported modality '{modality}'. Supported: {self.modalities}")

        batch_size, num_latents, input_dim = x.shape
        lambda_m = self.lambda_params[modality]  # [num_experts, output_dim]

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            lambda_i = lambda_m[i]  # [output_dim]
            weighted_output = expert_output * lambda_i
            expert_outputs.append(weighted_output)

        output = torch.stack(expert_outputs, dim=0).sum(dim=0)
        return output

class UnifiedEncoder(nn.Module):
    """Unified encoder for multiple neuroimaging modalities"""
    def __init__(self, in_dim=250, h=1024, out_dim=250, num_latents=128, qformer_spec=cfg.hyperparameter):
        super().__init__()
        self.fmri_subs = [i for i in range(1, 4)]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Initialize model components
        default_config = Config()
        self.encoder = Medformer(default_config)
        self.proj_neuro = nn.ModuleDict()
        self.projection = nn.ModuleDict()
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.moe_medformer = MoEMedformer(medformer_config=default_config)
        
        # Learnable weights
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_beta = nn.Parameter(torch.zeros(1))
        
        # Fusion heads for different modalities
        self.fusion_heads = nn.ModuleDict()
        self.modals = ['eeg', 'meg', 'fmri']
        
        # Initialize projection networks for each modality
        for modal in self.modals:
            proj_config = getattr(qformer_spec, modal).proj_neuro
            self.proj_neuro[modal] = Proj_neuro(
                emb_size=proj_config.emb_size,
                embedding_dim=proj_config.embedding_dim,
                proj_dim=proj_config.proj_dim,
                drop_proj=proj_config.drop_proj
            )
        
        # Initialize fusion heads using hydra
        for modal in self.modals:
            self.fusion_heads[modal] = hydra.utils.instantiate(getattr(qformer_spec, modal))
            self.fusion_heads[modal].init_cross_attn(qformer_spec, modal)
            
        # Positional embeddings and projections
        hid_width = 250
        self.conv1 = nn.ModuleDict()
        self.positional_embedding = nn.ParameterDict()
        self.num_voxels = {1: 6036, 2: 5944, 3: 5238}
        
        # Modality-specific initialization
        for modal in self.modals:
            if modal == 'eeg':
                modal_tokens = 63
                self.positional_embedding[modal] = nn.Parameter(torch.empty([modal_tokens, hid_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
                self.projection[modal] = nn.Upsample(size=256, mode='linear', align_corners=True)
                
            elif modal == 'meg':
                modal_tokens = 271
                self.conv1[modal] = nn.Upsample(size=250, mode='linear', align_corners=True)
                self.positional_embedding[modal] = nn.Parameter(torch.empty([modal_tokens, hid_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
                self.projection[modal] = nn.AdaptiveAvgPool1d(256)
                     
            elif modal == 'fmri':                
                self.conv1[modal] = nn.ModuleDict({
                    str(sub): nn.Linear(7000, 64000) for sub in self.fmri_subs
                })                
                self.positional_embedding[modal] = nn.Parameter(torch.empty([256, hid_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)

    def forward(self, x, subject_ids, modal):
        """Forward pass for unified encoder"""
        # Modality-specific preprocessing
        if modal == 'fmri':
            x = self.conv1[modal][f'{subject_ids[0].item()}'](x)                
        elif modal == 'meg':
            x = self.conv1[modal](x)

        # Reshape and add positional embeddings
        x = x.reshape(x.size(0), -1, 250)
        
        if modal in ['eeg', 'meg', 'fmri']:
            pos_embedding = self.positional_embedding[modal]
        
        x = x + pos_embedding.to(x.dtype)
        
        # Modality-specific projections
        if modal in ['eeg', 'meg']:
            x = x.transpose(1, 2)
            x = self.projection[modal](x)
            x = x.transpose(1, 2)
        
        # Process through MoE Medformer
        x = self.moe_medformer(x, modal)
        
        # EEG processing
        x = self.enc_eeg(x)
        x = self.proj_eeg(x)
        
        return x