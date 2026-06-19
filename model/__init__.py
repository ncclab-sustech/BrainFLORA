"""
BrainFLORA model package.

This package contains neural network models for multimodal brain signal encoding:

- Medformer: Transformer-based encoder for time series
- MedformerBase: Full-featured Medformer with multiple task support
- EEG/MEG/fMRI encoders: Modality-specific encoders
- UnifiedEncoder: Multimodal unified encoder
- DiffusionPrior: Diffusion-based prior model
"""

__all__ = [
    'Medformer',
    'MedformerBase',
    'Model',
    'FusionHead',
]


def __getattr__(name):
    if name in {'Medformer', 'MedformerBase', 'Model'}:
        from .Medformer import Medformer, MedformerBase, Model

        exports = {
            'Medformer': Medformer,
            'MedformerBase': MedformerBase,
            'Model': Model,
        }
        return exports[name]
    if name == 'FusionHead':
        from .projector import FusionHead

        return FusionHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
