"""Reusable Medformer encoders for EEG, MEG, and fMRI signals."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from model.Medformer import Medformer
from utils.losses import ClipLoss


class MedformerConfig:
    def __init__(self, seq_len: int, enc_in: int):
        self.task_name = "classification"
        self.seq_len = seq_len
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = "timeF"
        self.freq = "h"
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = "gelu"
        self.enc_in = enc_in
        self.single_channel = False
        self.patch_len_list = "2,4,8"
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"
        self.no_inter_attn = False
        self.num_class = 250


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FlattenHead(nn.Sequential):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class ConvPatchEmbedding(nn.Module):
    def __init__(self, spatial_kernel: int, emb_size: int = 40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (spatial_kernel, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        return self.projection(x)


class EEGNoTSPatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 1024):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.project = nn.Sequential(
            nn.Linear(55250, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.project(self.flatten(x))


class MEGNoTSPatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(250, 128),
            nn.BatchNorm1d(178),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, emb_size),
            nn.BatchNorm1d(178),
            nn.ELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_layers(x.transpose(1, 2).transpose(1, 2))


class fMRINoTSPatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40):
        super().__init__()
        flat_dim = 899 * 250
        self.linear_layers = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(flat_dim, 1440),
            nn.BatchNorm1d(1440),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.final_arrange = Rearrange("b (h w) -> b h w", h=36, w=40)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.linear_layers(x)
        return self.final_arrange(x)


class SignalHead(nn.Sequential):
    def __init__(self, patch_embedding: nn.Module):
        super().__init__(patch_embedding, FlattenHead())


class ProjectionHead(nn.Sequential):
    def __init__(self, embedding_dim: int = 1440, proj_dim: int = 1024, drop_proj: float = 0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class _BaseMedformerEncoder(nn.Module):
    def __init__(
        self,
        config: MedformerConfig,
        patch_embedding: nn.Module,
        embedding_dim: int = 1440,
        sequence_length: int = 250,
        num_subjects: int = 10,
    ):
        super().__init__()
        self.encoder = Medformer(config)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(config.d_model, sequence_length) for _ in range(num_subjects)]
        )
        self.enc_eeg = SignalHead(patch_embedding)
        self.proj_eeg = ProjectionHead(embedding_dim=embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def encode(self, x):
        x = self.encoder(x)
        x = self.enc_eeg(x)
        return self.proj_eeg(x)


class eeg_encoder(_BaseMedformerEncoder):
    def __init__(self, sequence_length: int = 250, num_subjects: int = 10, joint_train: bool = False):
        super().__init__(
            MedformerConfig(seq_len=250, enc_in=63),
            ConvPatchEmbedding(spatial_kernel=221),
            sequence_length=sequence_length,
            num_subjects=num_subjects,
        )

    def forward(self, x, subject_ids=None):
        return self.encode(x)


class meg_encoder(_BaseMedformerEncoder):
    def __init__(self, sequence_length: int = 250, num_subjects: int = 10, joint_train: bool = False):
        super().__init__(
            MedformerConfig(seq_len=201, enc_in=271),
            ConvPatchEmbedding(spatial_kernel=178),
            sequence_length=sequence_length,
            num_subjects=num_subjects,
        )

    def forward(self, x, subject_ids):
        return self.encode(x)


class fmri_encoder(_BaseMedformerEncoder):
    def __init__(self, sequence_length: int = 250, num_subjects: int = 10, joint_train: bool = False):
        super().__init__(
            MedformerConfig(seq_len=1024, enc_in=8),
            ConvPatchEmbedding(spatial_kernel=899),
            sequence_length=sequence_length,
            num_subjects=num_subjects,
        )
        self.fmri_subs = [i for i in range(1, 4)]
        self.num_voxels = {1: 6036, 2: 5944, 3: 5238}
        self.proj_fmri = nn.Linear(7000, 8192)

    def forward(self, x, subject_ids):
        x = self.proj_fmri(x)
        x = x.reshape(x.size(0), -1, 1024)
        return self.encode(x)


class eeg_encoder_nots(_BaseMedformerEncoder):
    def __init__(self, sequence_length: int = 250, num_subjects: int = 10, joint_train: bool = False):
        super().__init__(
            MedformerConfig(seq_len=250, enc_in=63),
            EEGNoTSPatchEmbedding(),
            sequence_length=sequence_length,
            num_subjects=num_subjects,
        )

    def forward(self, x, subject_ids=None):
        return self.encode(x)


class meg_encoder_nots(_BaseMedformerEncoder):
    def __init__(self, sequence_length: int = 250, num_subjects: int = 10, joint_train: bool = False):
        super().__init__(
            MedformerConfig(seq_len=201, enc_in=271),
            MEGNoTSPatchEmbedding(),
            embedding_dim=7120,
            sequence_length=sequence_length,
            num_subjects=num_subjects,
        )

    def forward(self, x, subject_ids):
        return self.encode(x)


class fmri_encoder_nots(_BaseMedformerEncoder):
    def __init__(self, sequence_length: int = 250, num_subjects: int = 10, joint_train: bool = False):
        super().__init__(
            MedformerConfig(seq_len=1024, enc_in=8),
            fMRINoTSPatchEmbedding(),
            sequence_length=sequence_length,
            num_subjects=num_subjects,
        )
        self.fmri_subs = [i for i in range(1, 4)]
        self.num_voxels = {1: 6036, 2: 5944, 3: 5238}
        self.proj_fmri = nn.Linear(7000, 8192)

    def forward(self, x, subject_ids):
        x = self.proj_fmri(x)
        x = x.reshape(x.size(0), -1, 1024)
        return self.encode(x)
