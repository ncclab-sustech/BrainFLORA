#!/usr/bin/env python
"""
Setup script for BrainFLORA project.
This allows the project to be installed as a package, enabling proper imports
without modifying sys.path.
"""

from setuptools import setup, find_packages

setup(
    name="brainflora",
    version="0.1.0",
    description="Reproducible multimodal neural embeddings for EEG, MEG, and fMRI decoding",
    author="BrainFLORA Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["configs", "imgs", "eval", "Retrieval"]),
    # Mark Retrieval and eval as namespace packages that can import from root
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # Core dependencies - can be loaded from requirements.txt
        "numpy",
        "matplotlib",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "transformers",
        "diffusers",
        "accelerate",
        "einops",
        "omegaconf",
        "reformer_pytorch",
        "tqdm",
        "clip",
        "open_clip_torch",
        "braindecode>=0.8.1",
        "wandb",
        "scikit-learn",
        "mne",
    ],
)
