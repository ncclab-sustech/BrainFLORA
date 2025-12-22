#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yml file for "conda env export > environment.yml" after running this.

set -e

conda create -n BrainFLORA python=3.10.8 -y
conda activate BrainFLORA

conda install numpy matplotlib tqdm scikit-image jupyterlab -y
conda install -c conda-forge accelerate -y

pip install clip-retrieval clip pandas matplotlib ftfy regex kornia umap-learn
pip install dalle2-pytorch

pip install open_clip_torch

pip install transformers==4.36.0
pip install diffusers==0.25.0

pip install braindecode==0.8.1
pip install accelerate==0.26.0
pip install mne-bids

pip install torchvision==0.15.2 torch==2.0.1

pip install info-nce-pytorch==0.1.0
pip install pytorch-msssim

pip install reformer_pytorch

pip install mne
pip install wandb
pip install einops