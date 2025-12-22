"""
BrainFLORA utilities package.

This package contains various utility modules:
- training: Training utilities (NativeScaler, wandb_logger, lr schedulers, etc.)
- losses: Loss functions (ClipLoss, SupConLoss, mixco_nce, etc.)
- misc: Miscellaneous utilities (distributed training helpers)
- metrics: Evaluation metrics
- masking: Data masking utilities
- tools: General helper tools
- timefeatures: Time feature extraction
"""

# Import commonly used utilities for convenience
from .training import (
    NativeScaler,
    wandb_logger,
    get_grad_norm_,
    train_one_epoch,
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    interpolate_pos_embed,
    adjust_learning_rate,
    load_model,
    patchify,
    unpatchify,
)

# Import loss functions
from .losses import (
    ClipLoss,
    SupConLoss,
    mixco_nce,
    mixco_1d,
    mixco_timeseries,
    soft_clip_loss,
    gather_features,
)

__all__ = [
    # Training utilities
    'NativeScaler',
    'wandb_logger',
    'get_grad_norm_',
    'train_one_epoch',
    'get_1d_sincos_pos_embed',
    'get_1d_sincos_pos_embed_from_grid',
    'interpolate_pos_embed',
    'adjust_learning_rate',
    'load_model',
    'patchify',
    'unpatchify',
    # Loss functions
    'ClipLoss',
    'SupConLoss',
    'mixco_nce',
    'mixco_1d',
    'mixco_timeseries',
    'soft_clip_loss',
    'gather_features',
]

