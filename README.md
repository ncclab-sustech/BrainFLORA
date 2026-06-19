<h1 align="center">
  BrainFLORA: Uncovering Brain Concept Representation<br>
  via Multimodal Neural Embeddings
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.09747"><img src="https://img.shields.io/badge/Paper-arXiv%3A2507.09747-B31B1B.svg" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/LidongYang/BrainFLORA"><img src="https://img.shields.io/badge/Data%20%26%20Checkpoints-Hugging%20Face-blue" alt="Hugging Face"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://dongyangli.site/">Dongyang Li</a>,
  <a href="https://github.com/wojiao-yc">Haoyang Qin</a>,
  <a href="https://www.wormforce.net/members/mingyang-wu">Mingyang Wu</a>,
  <a href="https://hedges0-0.github.io/">Chen Wei</a>,
  <a href="https://scholar.google.com/citations?user=UpP9hJ8AAAAJ&hl=en">Quanying Liu</a>
</p>

<p align="center">
  Southern University of Science and Technology
</p>

<p align="center">
  <img src="imgs/fig-overview_00.png" width="100%" alt="BrainFLORA overview">
</p>

<p align="center">
  <img src="imgs/fig-framework_00.png" width="100%" alt="BrainFLORA framework">
</p>

BrainFLORA is a reproducible multimodal neural embedding framework for EEG, MEG, and fMRI visual retrieval, image reconstruction, and image captioning. The repository provides unified command-line entry points for training, inference, and evaluation with the released checkpoints.

## News

- **2025-12-21**: Preprocessed datasets and pretrained checkpoints were released on [Hugging Face](https://huggingface.co/datasets/LidongYang/BrainFLORA).
- **2025-07-15**: The [arXiv paper](https://arxiv.org/abs/2507.09747) was released.
- **2025-07-12**: The codebase was released.
- **2025-07-05**: BrainFLORA was accepted by ACM MM 2025.

## Installation

```bash
git clone https://github.com/ncclab-sustech/BrainFLORA.git
cd BrainFLORA

conda env create -f environment.yml
conda activate BrainFLORA
pip install -e .
```

If you prefer the setup script from the original repository:

```bash
bash setup.sh
conda activate BrainFLORA
pip install -e .
```

Caption evaluation may require optional metric packages such as `clip` and `pycocoevalcap`. Caption generation uses Shikra; download the model assets from the original Shikra project and place them locally:

```text
external_models/shikra-7b
external_models/mm_projector.bin
```

Shikra resources:

- Paper: [Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic](https://arxiv.org/abs/2306.15195)
- Code and model instructions: [shikras/shikra](https://github.com/shikras/shikra)
- Hugging Face model page: [shikras/shikra-7b-delta-v1](https://huggingface.co/shikras/shikra-7b-delta-v1)

## Data and Checkpoints

The recommended path is to use the preprocessed data and released checkpoints from [LidongYang/BrainFLORA](https://huggingface.co/datasets/LidongYang/BrainFLORA). The reproduction scripts assume this layout by default:

```text
BrainFLORA/
  checkpoints/
    eeg_01-06_01-46_150.pth
    meg_01-11_14-50_150.pth
    fmri_01-18_01-35_150.pth
    Unified_EEG+MEG+fMRI_EEG_01-27_02-32_60.pth
    reconstruction_checkpoints/150.pth
    reconstruction_checkpoints/prior_diffusion/150.pth
    caption_checkpoints/90.pth
    caption_checkpoints/prior_diffusion/100.pth
  features/
  external_models/
```

Default dataset roots are under `/vePFS-0x0d/visual/dataset`. Override them when needed:

```bash
python eval/reproduce_retrieval.py \
  --eeg-data-path /path/to/eeg \
  --meg-data-path /path/to/meg \
  --fmri-data-path /path/to/fmri
```

Raw datasets used by the original project:

| Dataset | Link | Dataset | Link |
| --- | --- | --- | --- |
| THINGS-EEG1 | [OpenNeuro ds003825](https://openneuro.org/datasets/ds003825/versions/1.1.0) | THINGS-EEG2 | [OSF 3jk45](https://osf.io/3jk45/) |
| THINGS-MEG | [OpenNeuro ds004212](https://openneuro.org/datasets/ds004212/versions/2.0.0) | THINGS-fMRI | [OpenNeuro ds004192](https://openneuro.org/datasets/ds004192/versions/1.0.7) |
| THINGS-Images | [OSF rdxy2](https://osf.io/rdxy2) | | |

Use `data_preparing/` if you need to preprocess raw data yourself.

## Quick Run Experiments

The commands below replace the notebook-only evaluation commands from the original README. Use smoke runs first to verify paths and checkpoints, then run the full commands.

### 1. Visual Retrieval

Smoke run:

```bash
python eval/reproduce_retrieval.py \
  --device cuda:0 \
  --models single unified \
  --modalities fmri \
  --subjects sub-01 \
  --batch-size 32 \
  --output-dir outputs/reproduction_smoke
```

Full checkpoint evaluation:

```bash
python eval/reproduce_retrieval.py \
  --device cuda:0 \
  --models single unified \
  --modalities eeg meg fmri \
  --batch-size 128 \
  --output-dir outputs/reproduction
```

Outputs:

```text
outputs/reproduction/retrieval_reproduction_metrics.csv
outputs/reproduction/retrieval_reproduction_metrics.json
outputs/reproduction/retrieval_reproduction_vs_paper.json
```

### 2. Visual Reconstruction

High-level smoke run without SDXL image generation:

```bash
python eval/FLORA_inference_reconst.py \
  --device cuda:0 \
  --modalities fmri \
  --subjects sub-01 \
  --max-images 2 \
  --prior-steps 1 \
  --skip-image-generator \
  --output-dir outputs/reconstruction_highlevel_smoke \
  --summary-json outputs/reconstruction_highlevel_smoke/summary.json
```

Full image generation with cached SDXL/IP-Adapter assets:

```bash
python eval/FLORA_inference_reconst.py \
  --device cuda:0 \
  --modalities eeg meg fmri \
  --prior-steps 10 \
  --image-steps 4 \
  --local-files-only \
  --skip-existing \
  --output-dir outputs/reconstruction_png_full \
  --summary-json outputs/reconstruction_png_full/reconstruction_png_full_summary.json
```

The full run contains 3100 generated images:

```text
EEG: 10 subjects x 200 images
MEG: 4 subjects x 200 images
fMRI: 3 subjects x 100 images
```

Evaluate reconstruction metrics:

```bash
python eval/evaluate_reconstruction_metrics.py \
  --recon-root outputs/reconstruction_png_full \
  --modalities eeg meg fmri \
  --gt-root /path/to/dataset_root \
  --output-dir outputs/reconstruction_metrics
```

For a fast metric smoke test:

```bash
python eval/evaluate_reconstruction_metrics.py \
  --recon-root outputs/reconstruction_png_full \
  --modalities fmri \
  --subjects sub-01 \
  --gt-root /path/to/dataset_root \
  --metrics pixcorr ssim \
  --max-images 2 \
  --device cpu \
  --output-dir outputs/reconstruction_metrics_smoke
```

### 3. Visual Captioning

Generate woPrior caption token embeddings and apply the caption diffusion prior:

```bash
python eval/FLORA_inference_caption_embeddings.py \
  --device cuda:0 \
  --stage all \
  --modalities eeg meg fmri \
  --features-root features/FLORA \
  --metrics-json outputs/caption_embedding/caption_woPrior_metrics.json \
  --summary-json outputs/caption_embedding/caption_prior_summary.json
```

Generate caption JSON/TXT files with Shikra:

```bash
python eval/shikra_caption.py \
  --device cuda:0 \
  --modalities eeg meg fmri \
  --conditions Prior woPrior \
  --embeddings-root features/FLORA \
  --output-root Caption \
  --shikra-path external_models/shikra-7b \
  --mm-projector-path external_models/mm_projector.bin \
  --local-files-only
```

Dry-run Shikra caption generation without loading the model:

```bash
python eval/shikra_caption.py \
  --modalities fmri \
  --conditions Prior \
  --subjects sub-01 \
  --embeddings-root features/FLORA \
  --dry-run
```

Evaluate caption metrics:

```bash
python Caption/evaluate_caption_metrics.py \
  --modalities eeg fmri \
  --conditions Prior woPrior \
  --candidate-pattern 'shikra_{tag}_sub_{subject}_caption.json' \
  --output-dir outputs/caption_metrics
```

The default references are:

```text
Caption/EEG_caption/caption_EEG_GT.json
Caption/fMRI_caption/caption_fMRI_GT.json
```

## Quick Training

Unified training scripts are provided for retraining and ablations. Update dataset paths in the configs or pass CLI arguments before launching.

Train retrieval encoders:

```bash
# EEG
python Retrieval/train_retrieval.py \
  --modality eeg \
  --gpu cuda:0 \
  --output-dir outputs/contrast

# MEG
python Retrieval/train_retrieval.py \
  --modality meg \
  --gpu cuda:0 \
  --output-dir outputs/contrast

# fMRI
python Retrieval/train_retrieval.py \
  --modality fmri \
  --gpu cuda:0 \
  --output-dir outputs/contrast
```

Train the unified encoder for retrieval:

```bash
python train/train_unified_encoder.py \
  --task retrieval \
  --modalities eeg meg fmri \
  --gpu cuda:0 \
  --output_dir outputs/contrast
```

Train the unified encoder for reconstruction:

```bash
python train/train_unified_encoder.py \
  --task reconstruction \
  --modalities eeg meg fmri \
  --gpu cuda:0 \
  --output_dir outputs/contrast
```

Train the caption-aligned unified encoder:

```bash
python train/train_unified_encoder.py \
  --task caption \
  --use-caption \
  --modalities eeg meg fmri \
  --gpu cuda:0 \
  --output_dir outputs/contrast
```

Distributed reconstruction training:

```bash
accelerate launch train/train_unified_encoder.py \
  --task reconstruction \
  --distributed \
  --modalities eeg meg fmri \
  --output_dir outputs/contrast
```

## Repository Structure

```text
eval/                 Reproduction entry points for retrieval, reconstruction, and captioning
Caption/              Caption references and BrainHub metric wrapper
Retrieval/            Single-modality retrieval training entry point
train/                Unified encoder training scripts
model/, layers/       BrainFLORA model components
configs/              Training and inference configs
data_preparing/       Dataset loading and preprocessing code
utils/                Shared utilities
imgs/                 README figures
```

Large assets are ignored by Git and should be restored from releases or local cache:

```text
checkpoints/
features/
outputs/
external_models/
```

## Citation

```bibtex
@inproceedings{li2025brainflora,
  author = {Li, Dongyang and Qin, Haoyang and Wu, Mingyang and Wei, Chen and Liu, Quanying},
  title = {BrainFLORA: Uncovering Brain Concept Representation via Multimodal Neural Embeddings},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3754996},
  doi = {10.1145/3746027.3754996},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {5577--5586}
}

@article{li2024visual,
  title = {Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion},
  author = {Li, Dongyang and Wei, Chen and Li, Shiying and Zou, Jiachen and Liu, Quanying},
  journal = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {102822--102864},
  year = {2024}
}

@inproceedings{wei2024cocog,
  title = {CoCoG: controllable visual stimuli generation based on human concept representations},
  author = {Wei, Chen and Zou, Jiachen and Heinke, Dietmar and Liu, Quanying},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages = {3178--3186},
  year = {2024}
}
```

<!-- ## Acknowledge -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">😺Acknowledge</h2>

1.Thanks to Y Song et al. for their contribution in data set preprocessing and neural network structure, we refer to their work:"[Decoding Natural Images from EEG for Object Recognition](https://arxiv.org/pdf/2308.13234.pdf)". Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao.

2.We also thank the authors of [SDRecon](https://github.com/yu-takagi/StableDiffusionReconstruction) for providing the codes and the results. Some parts of the training script are based on [MindEye](https://medarc-ai.github.io/mindeye/) and [MindEye2](https://github.com/MedARC-AI/MindEyeV2). Thanks for the awesome research works.

3.Here we provide the THING-EEG2 dataset cited in the paper: "[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)". Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.


4.Another used THINGS-MEG and THINGS-fMRI data set provides a reference:"[THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior](https://elifesciences.org/articles/82580.pdf)". Hebart, Martin N., Oliver Contier, Lina Teichmann, Adam H. Rockter, Charles Y. Zheng, Alexis Kidder, Anna Corriveau, Maryam Vaziri-Pashkam, and Chris I. Baker.

5.We use the "[BrainHub](https://github.com/weihaox/BrainHub)" for visual caption evaluation from "[UMBRAE: Unified Multimodal Brain Decoding (ECCV 2024)](https://dl.acm.org/doi/abs/10.1007/978-3-031-72667-5_14)" Xia, Weihao and de Charette, Raoul and Oztireli, Cengiz and Xue, Jing-Hao.



Contact [Dongyang Li](https://github.com/dongyangli-del) if you have any questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.
