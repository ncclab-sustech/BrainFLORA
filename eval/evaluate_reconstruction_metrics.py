"""Evaluate BrainFLORA visual reconstruction outputs.

The metrics follow the original BrainFLORA reconstruction notebooks:
PixCorr, SSIM, two-way identification with AlexNet/Inception/CLIP, and
correlation distance with EfficientNet-B/SwAV.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_GT_ROOT = Path("/vePFS-0x0d/visual/dataset")
DEFAULT_GT_RELATIVE = {
    "eeg": Path("THINGS_EEG/images_set/test_images"),
    "meg": Path("THINGS_MEG/images_set_filter/test_images"),
    "fmri": Path("fmri_dataset/images/test_images"),
}
VALID_METRICS = [
    "pixcorr",
    "ssim",
    "alexnet2",
    "alexnet5",
    "inception",
    "clip",
    "effnet",
    "swav",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FeatureModel:
    model: Callable
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    feature_layer: str | None = None


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def subject_id(subject: str) -> str:
    match = re.search(r"\d+$", subject)
    if match is None:
        raise ValueError(f"Cannot parse subject id from {subject!r}")
    return f"sub-{int(match.group()):02d}"


def image_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def list_recon_images(subject_dir: Path) -> list[Path]:
    paths = sorted(subject_dir.glob("image_*.png"), key=image_sort_key)
    if not paths:
        raise FileNotFoundError(f"No reconstruction images found in {subject_dir}")
    return paths


def list_gt_images(gt_dir: Path, image_policy: str) -> list[Path]:
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT image directory does not exist: {gt_dir}")

    folders = sorted([p for p in gt_dir.iterdir() if p.is_dir()])
    if folders:
        images: list[Path] = []
        for folder in folders:
            folder_images = sorted(
                [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES],
                key=lambda p: p.name,
            )
            if not folder_images:
                continue
            if image_policy == "first-per-folder":
                images.append(folder_images[0])
            else:
                images.extend(folder_images)
        return images

    return sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)


def resolve_gt_dir(gt_root: Path, modality: str) -> Path:
    candidates = [
        gt_root / DEFAULT_GT_RELATIVE[modality],
        gt_root / modality / "test_images",
        gt_root / modality,
        gt_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_images(paths: list[Path], size: int) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((size, size), Image.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def resize_batch(images: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(images)


def compute_pixcorr(recons: torch.Tensor, targets: torch.Tensor) -> float:
    recons = resize_batch(recons, (425, 425)).flatten(1).numpy()
    targets = resize_batch(targets, (425, 425)).flatten(1).numpy()
    recon_centered = recons - recons.mean(axis=1, keepdims=True)
    target_centered = targets - targets.mean(axis=1, keepdims=True)
    numerator = (recon_centered * target_centered).sum(axis=1)
    denominator = np.linalg.norm(recon_centered, axis=1) * np.linalg.norm(target_centered, axis=1)
    return float(np.nanmean(numerator / np.maximum(denominator, 1e-12)))


def compute_ssim(recons: torch.Tensor, targets: torch.Tensor) -> float:
    recons = resize_batch(recons, 425).permute(0, 2, 3, 1).numpy()
    targets = resize_batch(targets, 425).permute(0, 2, 3, 1).numpy()
    recon_gray = rgb2gray(recons)
    target_gray = rgb2gray(targets)
    scores = [
        structural_similarity(
            rec,
            tgt,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
        for rec, tgt in zip(recon_gray, target_gray)
    ]
    return float(np.mean(scores))


def extract_features(
    images: torch.Tensor,
    feature_model: FeatureModel,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            batch = torch.stack([feature_model.preprocess(image) for image in batch], dim=0)
            feats = feature_model.model(batch)
            if feature_model.feature_layer is not None:
                feats = feats[feature_model.feature_layer]
            outputs.append(feats.float().flatten(1).cpu())
    return torch.cat(outputs, dim=0).numpy()


def two_way_identification(
    recons: torch.Tensor,
    targets: torch.Tensor,
    feature_model: FeatureModel,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    pred = extract_features(recons, feature_model, device=device, batch_size=batch_size)
    real = extract_features(targets, feature_model, device=device, batch_size=batch_size)
    pred = pred - pred.mean(axis=1, keepdims=True)
    real = real - real.mean(axis=1, keepdims=True)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    real = real / np.maximum(np.linalg.norm(real, axis=1, keepdims=True), 1e-12)
    correlations = real @ pred.T
    congruent = np.diag(correlations)
    success_count = (correlations < congruent[None, :]).sum(axis=0)
    return float(np.mean(success_count) / (len(targets) - 1))


def mean_correlation_distance(
    recons: torch.Tensor,
    targets: torch.Tensor,
    feature_model: FeatureModel,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    pred = extract_features(recons, feature_model, device=device, batch_size=batch_size)
    real = extract_features(targets, feature_model, device=device, batch_size=batch_size)
    pred = pred - pred.mean(axis=1, keepdims=True)
    real = real - real.mean(axis=1, keepdims=True)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    real = real / np.maximum(np.linalg.norm(real, axis=1, keepdims=True), 1e-12)
    return float(np.mean(1.0 - np.sum(pred * real, axis=1)))


def imagenet_preprocess(size: int, divide_by_255: bool = False) -> transforms.Compose:
    steps: list[Callable] = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)]
    if divide_by_255:
        steps.append(transforms.Lambda(lambda x: x.float() / 255.0))
    steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(steps)


def build_feature_models(metrics: set[str], device: torch.device) -> dict[str, FeatureModel]:
    models: dict[str, FeatureModel] = {}

    if {"alexnet2", "alexnet5"} & metrics:
        model = create_feature_extractor(
            alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
            return_nodes=["features.4", "features.11"],
        ).to(device)
        model.eval().requires_grad_(False)
        preprocess = imagenet_preprocess(256)
        models["alexnet2"] = FeatureModel(model, preprocess, "features.4")
        models["alexnet5"] = FeatureModel(model, preprocess, "features.11")

    if "inception" in metrics:
        model = create_feature_extractor(
            inception_v3(weights=Inception_V3_Weights.DEFAULT),
            return_nodes=["avgpool"],
        ).to(device)
        model.eval().requires_grad_(False)
        models["inception"] = FeatureModel(model, imagenet_preprocess(342), "avgpool")

    if "clip" in metrics:
        try:
            import clip
        except ImportError as exc:
            raise ImportError("CLIP metric requested but the `clip` package is not installed.") from exc

        model, _ = clip.load("ViT-L/14", device=device)
        model.eval().requires_grad_(False)
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        models["clip"] = FeatureModel(model.encode_image, preprocess, None)

    if "effnet" in metrics:
        model = create_feature_extractor(
            efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
            return_nodes=["avgpool"],
        ).to(device)
        model.eval().requires_grad_(False)
        models["effnet"] = FeatureModel(model, imagenet_preprocess(255), "avgpool")

    if "swav" in metrics:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        model = create_feature_extractor(model, return_nodes=["avgpool"]).to(device)
        model.eval().requires_grad_(False)
        models["swav"] = FeatureModel(model, imagenet_preprocess(224), "avgpool")

    return models


def evaluate_subject(
    *,
    modality: str,
    subject: str,
    recon_dir: Path,
    gt_images: list[Path],
    metrics: set[str],
    feature_models: dict[str, FeatureModel],
    device: torch.device,
    batch_size: int,
    image_size: int,
    max_images: int | None,
) -> dict[str, object]:
    recon_images = list_recon_images(recon_dir)
    n = min(len(recon_images), len(gt_images))
    if max_images is not None:
        n = min(n, max_images)
    if n < 2:
        raise ValueError(f"Need at least 2 paired images for {modality}/{subject}; got {n}.")
    if len(recon_images) != len(gt_images):
        print(
            f"Warning: {modality}/{subject} has {len(recon_images)} recon images and "
            f"{len(gt_images)} GT images; evaluating first {n}.",
            flush=True,
        )

    recon_images = recon_images[:n]
    gt_images = gt_images[:n]
    recons = load_images(recon_images, image_size)
    targets = load_images(gt_images, image_size)

    result: dict[str, object] = {
        "modality": modality,
        "subject": subject,
        "n_images": n,
        "recon_dir": str(recon_dir),
    }
    if "pixcorr" in metrics:
        result["PixCorr"] = compute_pixcorr(recons, targets)
    if "ssim" in metrics:
        result["SSIM"] = compute_ssim(recons, targets)
    if "alexnet2" in metrics:
        result["AlexNet(2)"] = two_way_identification(
            recons, targets, feature_models["alexnet2"], device=device, batch_size=batch_size
        )
    if "alexnet5" in metrics:
        result["AlexNet(5)"] = two_way_identification(
            recons, targets, feature_models["alexnet5"], device=device, batch_size=batch_size
        )
    if "inception" in metrics:
        result["InceptionV3"] = two_way_identification(
            recons, targets, feature_models["inception"], device=device, batch_size=batch_size
        )
    if "clip" in metrics:
        result["CLIP"] = two_way_identification(
            recons, targets, feature_models["clip"], device=device, batch_size=batch_size
        )
    if "effnet" in metrics:
        result["EffNet-B"] = mean_correlation_distance(
            recons, targets, feature_models["effnet"], device=device, batch_size=batch_size
        )
    if "swav" in metrics:
        result["SwAV"] = mean_correlation_distance(
            recons, targets, feature_models["swav"], device=device, batch_size=batch_size
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BrainFLORA reconstruction images.")
    parser.add_argument("--recon-root", type=Path, default=PROJECT_ROOT / "outputs/reconstruction_png_full")
    parser.add_argument("--modalities", nargs="+", choices=["eeg", "meg", "fmri"], default=["eeg", "meg", "fmri"])
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject subset, e.g. sub-01 02.")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reconstruction_metrics")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test image limit per subject.")
    parser.add_argument(
        "--gt-image-policy",
        choices=["first-per-folder", "all"],
        default="first-per-folder",
        help="How to expand GT image folders. BrainFLORA reconstruction uses first-per-folder by default.",
    )
    parser.add_argument("--metrics", nargs="+", choices=VALID_METRICS + ["all"], default=["all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recon_root = resolve_path(args.recon_root)
    gt_root = resolve_path(args.gt_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    metrics = set(VALID_METRICS if "all" in args.metrics else args.metrics)
    feature_models = build_feature_models(metrics, device)

    requested_subjects = [subject_id(s) for s in args.subjects] if args.subjects else None
    rows: list[dict[str, object]] = []
    for modality in args.modalities:
        modality_recon_root = recon_root / modality
        if not modality_recon_root.exists():
            raise FileNotFoundError(f"Missing reconstruction modality directory: {modality_recon_root}")
        subjects = requested_subjects or sorted(p.name for p in modality_recon_root.iterdir() if p.is_dir())
        gt_dir = resolve_gt_dir(gt_root, modality)
        gt_images = list_gt_images(gt_dir, args.gt_image_policy)
        print(f"{modality}: using {len(gt_images)} GT images from {gt_dir}", flush=True)

        for subject in subjects:
            subject_dir = modality_recon_root / subject
            if not subject_dir.exists():
                raise FileNotFoundError(f"Missing reconstruction subject directory: {subject_dir}")
            print(f"Evaluating modality={modality} subject={subject}", flush=True)
            row = evaluate_subject(
                modality=modality,
                subject=subject,
                recon_dir=subject_dir,
                gt_images=gt_images,
                metrics=metrics,
                feature_models=feature_models,
                device=device,
                batch_size=args.batch_size,
                image_size=args.image_size,
                max_images=args.max_images,
            )
            rows.append(row)
            print(pd.DataFrame([row]).to_string(index=False), flush=True)

    df = pd.DataFrame(rows)
    numeric_cols = [col for col in df.columns if col not in {"modality", "subject", "recon_dir"}]
    modality_avg = df.groupby("modality", as_index=False)[numeric_cols].mean(numeric_only=True)
    modality_avg.insert(1, "subject", "Average")
    overall_avg = pd.DataFrame([{**{"modality": "all", "subject": "Average"}, **df[numeric_cols].mean(numeric_only=True).to_dict()}])
    summary = pd.concat([df, modality_avg, overall_avg], ignore_index=True, sort=False)

    per_subject_csv = output_dir / "reconstruction_metrics_per_subject.csv"
    summary_csv = output_dir / "reconstruction_metrics_summary.csv"
    per_subject_json = output_dir / "reconstruction_metrics_per_subject.json"
    summary_json = output_dir / "reconstruction_metrics_summary.json"
    df.to_csv(per_subject_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    per_subject_json.write_text(json.dumps(rows, indent=2) + "\n")
    summary_json.write_text(summary.to_json(orient="records", indent=2) + "\n")

    print(f"Wrote {per_subject_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_subject_json}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
