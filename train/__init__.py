"""
BrainFLORA training scripts package.

This package contains training scripts for the unified encoder:
- train_unified_encoder: Basic unified encoder training
- train_unified_encoder_highlevel_diffprior: Training with diffusion prior
- train_unified_encoder_highlevel_diffprior_parallel: Parallel training with diffusion prior
- train_unified_encoder_highlevel_diffprior_caption: Training with caption support
"""

__all__ = [
    'train_unified_encoder',
    'train_unified_encoder_highlevel_diffprior',
    'train_unified_encoder_highlevel_diffprior_parallel',
    'train_unified_encoder_highlevel_diffprior_caption',
]

