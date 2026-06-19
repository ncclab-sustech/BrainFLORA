# BrainHub Caption Metrics Adapter

This directory contains a local BrainFLORA adapter of the caption-evaluation
subset from the external [BrainHub](https://github.com/weihaox/BrainHub)
benchmark used by [UMBRAE](https://weihaox.github.io/UMBRAE/).

The original BrainHub project is authored by Weihao Xia and released under the
MIT license. The license is preserved in `LICENSE`. BrainFLORA keeps only the
caption metric code required for paper reproduction and has adjusted local
paths/imports for use through `Caption/evaluate_caption_metrics.py`.

## Included Functionality

- `eval_caption.py`: evaluates generated captions with CLIPScore,
  RefCLIPScore, and optional reference-based caption metrics.
- `metrics.py`: wraps BLEU, METEOR, ROUGE, CIDEr, and SPICE via
  `pycocoevalcap`.
- `run.sh`: original-style examples retained as reference only.

The full BrainHub benchmark also includes grounding evaluation assets and
leaderboard files. Those assets are not bundled here; use the upstream
BrainHub repository for full benchmark reproduction.

## Expected Inputs

`eval_caption.py` expects:

```bash
python eval_caption.py <candidates_json> <image_dir> --references_json <references_json>
```

where:

- `<candidates_json>` maps image ids to generated captions.
- `<references_json>` maps the same image ids to one or more reference captions.
- `<image_dir>` contains images whose stems match the image ids.

For BrainFLORA experiments, prefer the repository-level wrapper:

```bash
python Caption/evaluate_caption_metrics.py \
  --modalities eeg meg fmri \
  --conditions Prior woPrior \
  --skip-missing
```

## Optional Dependencies

Caption metrics require optional packages that may not be installed in a
minimal environment:

- `clip`
- `pycocoevalcap`
- Java runtime for METEOR/SPICE in some `pycocoevalcap` installations

If these dependencies are missing, the BrainFLORA wrapper will fail at metric
execution time rather than silently reporting incomplete scores.
