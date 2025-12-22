<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">🧠BrainFLORA: Uncovering Brain Concept Representation via Multimodal Neural Embeddings</h2>
</div>


<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">

<p align="center">
  <a href="#">
  <p align="center">
    <a href='https://arxiv.org/abs/2507.09747'><img src='http://img.shields.io/badge/Paper-arxiv.2403.07721-B31B1B.svg'></a>
    <a href='https://huggingface.co/datasets/LidongYang/BrainFLORA'><img src='https://img.shields.io/badge/BrainFLORA-%F0%9F%A4%97%20Hugging%20Face-blue'></a>
  </p>
</p>


</div>

<br/>


<div align="center">
<!--  -->
<div>
<img src="imgs/fig-overview_00.png" alt="fig-genexample" style="max-width: 75%; height: auto;"/>  
</div>

</div>

A comparative overview of multimodal decoding paradigms.

<div align="center">
<div>
<img src="imgs/fig-framework_00.png" alt="Framework" style="max-width: 70%; height: auto;"/>
</div>
</div>

Overall architecture of BrainFLORA.




<!-- ## News -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">🐣 Update</h2>

* **2025/12/21**, we released the preprocessed datasets and pretrained checkpoints on [🤗 Hugging Face](https://huggingface.co/datasets/LidongYang/BrainFLORA).
* **2025/07/15**, the [arxiv](https://arxiv.org/abs/2507.09747) paper is public.
* **2025/07/12**, we officially released the code.
* **2025/07/05**, BrainFLORA is accepted by *ACM MM 2025*.


<!-- ## Environment setup -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">🛠️ Environment Setup</h2>

### Quick Start

**Option 1: Using setup script (Recommended)**
```bash
bash setup.sh
conda activate BrainFLORA
```

**Option 2: Using conda environment file**
```bash
conda env create -f environment.yml
conda activate BrainFLORA
```

**Option 3: Using pip**
```bash
pip install -r requirements.txt
```

**Important: Install as editable package**

After setting up the environment using any of the above options, install the project in editable mode to enable proper module imports:
```bash
pip install -e .
```

<!-- ## Prepare for Dataset -->

<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">📊 Dataset Preparation</h2>

### Option 1: Using Preprocessed Data (Recommended)

We provide preprocessed datasets ready for training on Hugging Face:

```python
from datasets import load_dataset
# Load the preprocessed BrainFLORA dataset
dataset = load_dataset("LidongYang/BrainFLORA")
```


👉 **[Download from Hugging Face](https://huggingface.co/datasets/LidongYang/BrainFLORA)**

### Option 2: Download Raw Data

To download and preprocess the raw data yourself:

Dataset | Download path| Dataset | Download path
:---: | :---:|:---: | :---:
THINGS-EEG1 |  [Download](https://openneuro.org/datasets/ds003825/versions/1.1.0) | THINGS-EEG2 | [Download](https://osf.io/3jk45/)
THINGS-MEG |  [Download](https://openneuro.org/datasets/ds004212/versions/2.0.0)| THINGS-fMRI  |  [Download](https://openneuro.org/datasets/ds004192/versions/1.0.7)
THINGS-Images |  [Download](https://osf.io/rdxy2)

After downloading, use the preprocessing scripts in `data_preparing/` directory to process the raw data.


<!-- ## Quick training and test  -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">🚴‍♂️Quick Training</h2>

#### 1. Visual Retrieval
We provide the script to train the modality encoders for ``joint subject training`` in *THINGS-EEG2* dataset. Please modify your dataset path and run:
```bash
python Retrieval/retrieval_joint_train_medformer.py --logger True --gpu cuda:0 --output_dir ./outputs/contrast
```

Additionally, replicate the results of other modalities (e.g. MEG, fMRI) by running:
```bash
# MEG
python Retrieval/retrieval_joint_train_MEG_medformer.py --logger True --gpu cuda:0 --output_dir ./outputs/contrast

# fMRI
python Retrieval/retrieval_joint_train_fMRI_medformer.py --logger True --gpu cuda:0 --output_dir ./outputs/contrast
```

#### 2. Visual Reconstruction
We provide quick training and inference scripts for ``high level and low level pipeline`` of visual reconstruction. Please modify your dataset path and run:
```bash
# Train and get multimodal neural embeddings aligned with CLIP embedding:
python train/train_unified_encoder_highlevel_diffprior.py \
    --modalities eeg meg fmri \
    --gpu cuda:0 \
    --output_dir ./outputs/contrast
```

#### 3. Visual Captioning
We provide scripts for visual caption generation:
```bash
# Train feature adapter with caption support
python train/train_unified_encoder_highlevel_diffprior_caption.py \
    --modalities eeg meg fmri \
    --gpu cuda:0 \
    --output_dir ./outputs/contrast
```

#### 4. Distributed Training (Multi-GPU)
For multi-GPU training with accelerate:
```bash
accelerate launch train/train_unified_encoder_highlevel_diffprior_parallel.py \
    --modalities eeg meg fmri \
    --output_dir ./outputs/contrast
```

<!-- ## Quick training and test  -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">🚴‍♂️Quick Evaluation</h2>

#### 1.Visual Retrieval
We provide the script to evaluation the models:
```
cd eval/
FLORA_inference.ipynb
```


#### 2.Visual Reconstruction

```
# Reconstruct images by assigning modalities and subjects:
cd eval/
python FLORA_inference_reconst.py
```



#### 3.Visual Captioning

```
# Get captions from prior latent
cd eval/
FLORA_inference_caption.ipynb
```

<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">👍 Citations</h2>

If you find our work useful, please consider citing:


```
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
  pages = {5577–5586}
}

@article{li2024visual,
  title={Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion},
  author={Li, Dongyang and Wei, Chen and Li, Shiying and Zou, Jiachen and Liu, Quanying},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={102822--102864},
  year={2024}
}
@inproceedings{wei2024cocog,
  title={CoCoG: controllable visual stimuli generation based on human concep08/03/2024t representations},
  author={Wei, Chen and Zou, Jiachen and Heinke, Dietmar and Liu, Quanying},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={3178--3186},
  year={2024}
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

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.