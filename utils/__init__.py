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

_TRAINING_EXPORTS = {
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
}

_LOSS_EXPORTS = {
    'ClipLoss',
    'SupConLoss',
    'mixco_nce',
    'mixco_1d',
    'mixco_timeseries',
    'soft_clip_loss',
    'gather_features',
}


def __getattr__(name):
    if name in _TRAINING_EXPORTS:
        from . import training

        return getattr(training, name)
    if name in _LOSS_EXPORTS:
        from . import losses

        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
