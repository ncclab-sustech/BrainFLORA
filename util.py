"""
Backward compatibility module.

This module re-exports utilities from utils.training for backward compatibility.
New code should import directly from utils or utils.training.

Example:
    # Old way (still works)
    from util import wandb_logger
    
    # New way (preferred)
    from utils import wandb_logger
    # or
    from utils.training import wandb_logger
"""

# Re-export everything from utils.training for backward compatibility
from utils.training import (
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

__all__ = [
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
]
