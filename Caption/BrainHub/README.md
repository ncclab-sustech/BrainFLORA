# BrainHub: Multimodal Brain Understanding Benchmark

## Updates
- [2024/07/01] UMBRAE is accepted to ECCV 2024.
- [2024/05/18] Update Leaderboard results.
- [2024/04/11] The brainhub benchmark has been released.

## Motivation

Unlike texts, images, or audio, whose contents are intuitively aligned with human perception and judgment, we lack sufficient knowledge of the information contained in captured brain responses, as they are not directly interpretable or interoperable to humans. We could translate the brain's responses into other understandable modalities as an indirect method of ascertaining its ability to describe, recognize, and localize instances, as well as discern spatial relationships among multiple exemplars. These abilities are important for brain-machine interfaces and other brain-related research. Therefore, we construct BrainHub, a brain understanding benchmark, based on [NSD](https://naturalscenesdataset.org/) and [COCO](https://cocodataset.org).

## Tasks and Metrics

The objectives are categorized into concept recognition and spatial localization, including:

- brain captioning, which is to generate textual descriptions summarizing the primary content of a given brain response. To evaluate the quality of generated captions, we use five [standard metrics](https://github.com/tylin/coco-caption), BLEU, METEOR, ROUGE, CIDEr, and SPICE, in addition to [CLIP-based scores](https://github.com/jmhessel/clipscore), CLIP-S and RefCLIP-S.

- brain grounding, which is the counterpart of visual grounding and seeks to recover spatial locations from brain responses by inferring coordinates. Given that identified classes might be named differently, or simply absent from ground truth labels, we evaluate boundingboxes through REC, using accuracy and IoU as metrics.

## Evaluation

There are 982 test images, 80 classes, 4,913 captions, and 5,829 boundingboxes. For grounding evaluation, we further group the 80 classes of COCO into four salience categories according to their salience in images: Salient (S), Salient Creatures (SC), Salient Objects (SO), and Inconspicuous (I). The illustration shows the statistics and mapping of our categories, w.r.t. COCO classes.

We provide the processed [text](https://github.com/weihaox/brainhub/caption) and [boundingbox](https://github.com/weihaox/brainhub/bbox) groundtruth. The demo evaluation script is provided [here](https://github.com/weihaox/brainhub/run.sh). If you would like to evaluate your produced results, please modify the result path accordingly.

We also provide baseline results associated with BrainHub, including the captioning results from [SDRecon](https://github.com/yu-takagi/StableDiffusionReconstruction), [BrainCap](https://arxiv.org/abs/2305.11560), and [OneLLM](https://onellm.csuhan.com/), as well as the captioning and grounding results from [UMBRAE](https://weihaox.github.io/UMBRAE/).

For contributing, please (a) update the leaderboard and (b) upload the results to the desired path with the required file name, such as `caption/comparison/umbrae/sub01_decoded_caption.json`.

## Leaderboard

### Captioning

This is the quantitative comparison for subject 1 (S1). For results on other subjects, refer to the [paper](https://weihaox.github.io/UMBRAE/). 'UMBRAE-S1' refers to model trained with S1 only, while 'UMBRAE' denotes the model with cross-subject training.

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S1   | 59.44 | 19.03 | 19.45  | 43.71 | 61.06 | 12.79 | 67.78 | 73.54    |
| UMBRAE-S1 | S1   | 57.63 | 16.76 | 18.41  | 42.15 | 51.93 | 11.83 | 66.44 | 72.12    |
| BrainCap  | S1   | 55.96 | 14.51 | 16.68  | 40.69 | 41.30 | 9.06  | 64.31 | 69.90    |
| OneLLM    | S1   | 47.04 | 9.51  | 13.55  | 35.05 | 22.99 | 6.26  | 54.80 | 61.28    |
| SDRecon   | S1   | 36.21 | 3.43  | 10.03  | 25.13 | 13.83 | 5.02  | 61.07 | 66.36    |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S2   | 59.37 | 18.41 | 19.17  | 43.86 | 55.93 | 12.08 | 66.46 | 72.36    |
| UMBRAE-S2 | S2   | 57.18 | 17.18 | 18.11  | 41.85 | 50.62 | 11.50 | 64.87 | 71.06    |
| BrainCap  | S2   | 53.80 | 13.03 | 15.90  | 39.96 | 35.60 | 8.47  | 62.48 | 68.19    |
| SDRecon   | S2   | 34.71 | 3.02  | 9.60   | 24.22 | 13.38 | 4.58  | 59.52 | 65.30    |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S5   | 60.36 | 19.03 | 20.04  | 44.81 | 61.32 | 13.19 | 68.39 | 74.11    |
| UMBRAE-S5 | S5   | 58.99 | 18.73 | 19.04  | 43.30 | 57.09 | 12.70 | 66.48 | 72.69    |
| BrainCap  | S5   | 55.28 | 14.62 | 16.45  | 40.87 | 41.05 | 9.24  | 63.89 | 69.64    |
| SDRecon   | S5   | 34.96 | 3.49  | 9.93   | 24.77 | 13.85 | 5.19  | 60.83 | 66.30    |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S7   | 57.20 | 17.13 | 18.29  | 42.16 | 52.73 | 11.63 | 65.90 | 71.83    |
| UMBRAE-S7 | S7   | 55.71 | 15.75 | 17.51  | 40.64 | 47.07 | 11.26 | 63.66 | 70.09    |
| BrainCap  | S7   | 54.25 | 14.00 | 15.94  | 40.02 | 37.49 | 8.57  | 62.52 | 68.48    |
| SDRecon   | S7   | 34.99 | 3.26  | 9.54   | 24.33 | 13.01 | 4.74  | 58.68 | 64.59    |

### Grounding

| Method    | Eval | acc@0.5 (A) | IoU (A) | acc@0.5 (S) | IoU (S) | acc@0.5 (I) | IoU (I) |
|-----------|------|-------------|---------|-------------|---------|-------------|---------|
| UMBRAE    | S1   | 18.93       | 21.28   | 30.23       | 30.18   | 4.83        | 10.18   |
| UMBRAE-S1 | S1   | 13.72       | 17.56   | 21.52       | 25.14   | 4.00        | 8.08    |
| UMBRAE    | S2   | 18.27       | 20.77   | 28.22       | 29.19   | 5.86        | 1025    |
| UMBRAE-S2 | S2   | 15.21       | 18.68   | 23.60       | 26.59   | 4.74        | 8.81    |
| UMBRAE    | S5   | 18.19       | 20.85   | 28.74       | 30.02   | 5.02        | 9.41    |
| UMBRAE-S5 | S5   | 14.72       | 18.45   | 22.93       | 26.34   | 4.46        | 8.60    |
| UMBRAE    | S7   | 16.74       | 19.63   | 25.69       | 27.90   | 5.58        | 9.31    |
| UMBRAE-S7 | S7   | 13.60       | 17.83   | 21.07       | 25.19   | 4.28        | 8.64    |

## Citation

```bibtex
@inproceedings{xia2024umbrae,
  author    = {Xia, Weihao and de Charette, Raoul and Öztireli, Cengiz and Xue, Jing-Hao},
  title     = {UMBRAE: Unified Multimodal Brain Decoding},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```