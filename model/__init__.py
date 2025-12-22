"""
BrainFLORA model package.

This package contains neural network models for multimodal brain signal encoding:

- Medformer: Transformer-based encoder for time series
- MedformerBase: Full-featured Medformer with multiple task support
- EEG/MEG/fMRI encoders: Modality-specific encoders
- UnifiedEncoder: Multimodal unified encoder
- DiffusionPrior: Diffusion-based prior model
"""

from .Medformer import Medformer, MedformerBase, Model
from .projector import FusionHead

__all__ = [
    'Medformer',
    'MedformerBase', 
    'Model',
    'FusionHead',
]

