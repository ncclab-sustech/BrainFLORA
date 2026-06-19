"""Reproduce BrainFLORA visual retrieval metrics.

This script evaluates single-modality encoders and the unified EEG+MEG+fMRI
retrieval model from pretrained checkpoints. It writes subject-level and mean
metrics to CSV/JSON and, for the default metric set, writes a paper-comparison
JSON used by the README reproducibility table.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preparing import eegdatasets as eeg_dataset_module
from data_preparing import fmri_datasets_joint_subjects as fmri_dataset_module
from data_preparing import megdatasets_averaged as meg_dataset_module
from data_preparing.eegdatasets import EEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.megdatasets_averaged import MEGDataset
from model.EEG_MedformerTS import eeg_encoder
from model.MEG_MedformerTS import meg_encoder
from model.fMRI_MedformerTS import fmri_encoder
from model.unified_encoder_multi_tower import UnifiedEncoder
from utils.losses import ClipLoss


DEFAULT_DATA_PATHS = {
    "eeg": Path("/vePFS-0x0d/visual/dataset/THINGS_EEG/Preprocessed_data_250Hz"),
    "meg": Path("/vePFS-0x0d/visual/dataset/THINGS_MEG/preprocessed_newsplit"),
    "fmri": Path("/vePFS-0x0d/visual/dataset/fmri_dataset/Preprocessed"),
}

DEFAULT_TEST_IMAGE_DIRS = {
    "eeg": Path("/vePFS-0x0d/visual/dataset/THINGS_EEG/images_set/test_images"),
    "meg": Path("/vePFS-0x0d/visual/dataset/THINGS_MEG/images_set_filter/test_images"),
    "fmri": Path("/vePFS-0x0d/visual/dataset/fmri_dataset/images/test_images"),
}

DEFAULT_SINGLE_CHECKPOINTS = {
    "eeg": PROJECT_ROOT / "checkpoints/eeg_01-06_01-46_150.pth",
    "meg": PROJECT_ROOT / "checkpoints/meg_01-11_14-50_150.pth",
    "fmri": PROJECT_ROOT / "checkpoints/fmri_01-18_01-35_150.pth",
}

DEFAULT_UNIFIED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/Unified_EEG+MEG+fMRI_EEG_01-27_02-32_60.pth"
)

SUBJECTS = {
    "eeg": [f"sub-{i:02d}" for i in range(1, 11)],
    "meg": [f"sub-{i:02d}" for i in range(1, 5)],
    "fmri": [f"sub-{i:02d}" for i in range(1, 4)],
}

PAPER_RETRIEVAL_METRICS = {
    ("single", "eeg"): {
        "paper_name": "BrainFLORA-uni",
        "acc@2": 95.55,
        "acc@4": 86.90,
        "acc@10": 73.45,
        "acc@200": 25.35,
        "top5@200": 57.30,
    },
    ("single", "fmri"): {
        "paper_name": "BrainFLORA-uni",
        "acc@2": 91.33,
        "acc@4": 79.00,
        "acc@10": 63.00,
        "acc@100": 26.33,
        "top5@100": 57.33,
    },
    ("single", "meg"): {
        "paper_name": "BrainFLORA-unimodal / Table 5",
        "acc@2": 81.75,
        "acc@4": 64.50,
        "acc@10": 46.62,
        "acc@200": 8.00,
        "top5@200": 24.38,
    },
    ("unified", "eeg"): {
        "paper_name": "BrainFLORA-multi",
        "acc@2": 94.05,
        "acc@4": 87.30,
        "acc@10": 73.15,
        "acc@200": 25.05,
        "top5@200": 56.35,
    },
    ("unified", "fmri"): {
        "paper_name": "BrainFLORA-multi",
        "acc@2": 92.33,
        "acc@4": 84.67,
        "acc@10": 70.67,
        "acc@100": 28.33,
        "top5@100": 63.33,
    },
    ("unified", "meg"): {
        "paper_name": "BrainFLORA-multimodal / Table 5",
        "acc@2": 80.50,
        "acc@4": 61.88,
        "acc@10": 39.75,
        "acc@200": 6.88,
        "top5@200": 23.38,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_number(subject: str) -> int:
    match = re.search(r"\d+$", subject)
    if match is None:
        raise ValueError(f"Cannot parse subject id from {subject!r}")
    return int(match.group())


def normalize_subject(subject: str) -> str:
    return f"sub-{subject_number(subject):02d}"


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def torch_load(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def configure_dataset_image_dirs(image_dirs: dict[str, Path]) -> None:
    eeg_dataset_module.img_directory_test = str(image_dirs["eeg"])
    meg_dataset_module.img_directory_test = str(image_dirs["meg"])
    fmri_dataset_module.img_directory_test = str(image_dirs["fmri"])


def build_dataset(modality: str, subject: str, data_paths: dict[str, Path]):
    if modality == "eeg":
        return EEGDataset(str(data_paths["eeg"]), subjects=[subject], train=False)
    if modality == "meg":
        return MEGDataset(str(data_paths["meg"]), subjects=[subject], train=False)
    if modality == "fmri":
        return fMRIDataset(str(data_paths["fmri"]), adap_subject=subject, subjects=[subject], train=False)
    raise ValueError(f"Unknown modality: {modality}")


def candidate_img_features(dataset, modality: str) -> torch.Tensor:
    if modality == "meg":
        return dataset.img_features[::12]
    return dataset.img_features


def build_single_model(
    modality: str,
    device: torch.device,
    single_checkpoints: dict[str, Path],
):
    model_cls = {"eeg": eeg_encoder, "meg": meg_encoder, "fmri": fmri_encoder}[modality]
    model = model_cls()
    require_path(single_checkpoints[modality], f"{modality} checkpoint")
    state = torch_load(single_checkpoints[modality], map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_unified_model(
    device: torch.device,
    single_checkpoints: dict[str, Path],
    unified_checkpoint: Path,
):
    for modality, checkpoint in single_checkpoints.items():
        require_path(checkpoint, f"{modality} checkpoint")
    require_path(unified_checkpoint, "unified checkpoint")

    model = UnifiedEncoder(
        {modality: str(path) for modality, path in single_checkpoints.items()},
        device=device,
        num_experts=5,
        num_heads=2,
        ff_dim=128,
        num_layers=2,
    )
    state = torch_load(unified_checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def forward_model(model, data, subject_ids, modality: str, model_kind: str):
    if model_kind == "unified":
        return model(data, subject_ids, modal=modality).float()
    return model(data, subject_ids).float()


@torch.no_grad()
def evaluate_subject(
    model,
    model_kind: str,
    modality: str,
    subject: str,
    device: torch.device,
    batch_size: int,
    k_values: list[int],
    data_paths: dict[str, Path],
):
    dataset = build_dataset(modality, subject, data_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    img_features_all = candidate_img_features(dataset, modality).to(device).float()
    n_classes = img_features_all.shape[0]
    valid_k_values = [k for k in k_values if k <= n_classes]
    all_labels = set(range(n_classes))
    loss_func = ClipLoss()
    logit_scale = model.logit_scale.float()

    totals = {k: 0 for k in valid_k_values}
    correct = {k: 0 for k in valid_k_values}
    top5_correct = {k: 0 for k in valid_k_values}
    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    for batch in loader:
        _, data, labels, _, text_features, _, img_features, _, _, _ = batch
        data = data.to(device).float()
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        text_features = text_features.to(device).float()
        subject_ids = torch.full(
            (data.shape[0],),
            subject_number(subject),
            dtype=torch.long,
            device=device,
        )

        neural_features = forward_model(model, data, subject_ids, modality, model_kind)
        img_loss = loss_func(neural_features, img_features, logit_scale)
        text_loss = loss_func(neural_features, text_features, logit_scale)
        total_loss += (0.99 * img_loss + 0.01 * text_loss).item()
        total_batches += 1
        total_samples += labels.numel()

        for row, label_tensor in enumerate(labels):
            label = int(label_tensor.item())
            possible = list(all_labels - {label})
            for k in valid_k_values:
                selected = random.sample(possible, k - 1) + [label]
                selected_features = img_features_all[selected]
                logits = logit_scale * neural_features[row] @ selected_features.T
                predicted = selected[int(torch.argmax(logits).item())]
                totals[k] += 1
                correct[k] += int(predicted == label)
                if k >= 5:
                    _, top_indices = torch.topk(logits, 5, largest=True)
                    top5_labels = [selected[i] for i in top_indices.tolist()]
                    top5_correct[k] += int(label in top5_labels)

    result: dict[str, Any] = {
        "model": model_kind,
        "modality": modality,
        "subject": subject,
        "n_classes": n_classes,
        "n_samples": total_samples,
        "loss": total_loss / max(total_batches, 1),
    }
    for k in valid_k_values:
        result[f"acc@{k}"] = correct[k] / max(totals[k], 1)
        if k >= 5:
            result[f"top5@{k}"] = top5_correct[k] / max(totals[k], 1)
    return result


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["modality"]), []).append(row)

    summaries = []
    for (model, modality), items in sorted(groups.items()):
        summary: dict[str, Any] = {
            "model": model,
            "modality": modality,
            "subject": "mean",
            "n_classes": items[0]["n_classes"],
            "n_samples": sum(item["n_samples"] for item in items),
        }
        metric_keys = sorted(k for k in items[0] if k.startswith(("acc@", "top5@", "loss")))
        for key in metric_keys:
            summary[key] = float(np.mean([item[key] for item in items if key in item]))
        summaries.append(summary)
    return summaries


def paper_comparison(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        key = (summary["model"], summary["modality"])
        paper = PAPER_RETRIEVAL_METRICS.get(key)
        if paper is None:
            continue
        row: dict[str, Any] = {
            "model": summary["model"],
            "modality": summary["modality"],
            "paper_name": paper["paper_name"],
        }
        for metric, paper_pct in paper.items():
            if metric == "paper_name" or metric not in summary:
                continue
            local_pct = round(float(summary[metric]) * 100.0, 2)
            row[metric] = {
                "local_pct": local_pct,
                "paper_pct": float(paper_pct),
                "delta_pct": round(local_pct - float(paper_pct), 2),
            }
        rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, write_paper_comparison: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = aggregate(rows)
    all_rows = rows + summaries
    keys = sorted({key for row in all_rows for key in row})
    csv_path = output_dir / "retrieval_reproduction_metrics.csv"
    json_path = output_dir / "retrieval_reproduction_metrics.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps({"subjects": rows, "summary": summaries}, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")

    if write_paper_comparison:
        comparison_path = output_dir / "retrieval_reproduction_vs_paper.json"
        comparison = paper_comparison(summaries)
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
        print(f"Wrote {comparison_path}")
    print(json.dumps({"summary": summaries}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce BrainFLORA retrieval metrics.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["single", "unified"], default=["single", "unified"])
    parser.add_argument("--modalities", nargs="+", choices=["eeg", "meg", "fmri"], default=["eeg", "meg", "fmri"])
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset, e.g. sub-01 02")
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 10, 50, 100, 200])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reproduction")
    parser.add_argument("--eeg-data-path", type=Path, default=DEFAULT_DATA_PATHS["eeg"])
    parser.add_argument("--meg-data-path", type=Path, default=DEFAULT_DATA_PATHS["meg"])
    parser.add_argument("--fmri-data-path", type=Path, default=DEFAULT_DATA_PATHS["fmri"])
    parser.add_argument("--eeg-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["eeg"])
    parser.add_argument("--meg-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["meg"])
    parser.add_argument("--fmri-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["fmri"])
    parser.add_argument("--eeg-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["eeg"])
    parser.add_argument("--meg-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["meg"])
    parser.add_argument("--fmri-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["fmri"])
    parser.add_argument("--unified-checkpoint", type=Path, default=DEFAULT_UNIFIED_CHECKPOINT)
    parser.add_argument(
        "--no-paper-comparison",
        action="store_true",
        help="Do not write retrieval_reproduction_vs_paper.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data_paths = {
        "eeg": resolve_path(args.eeg_data_path),
        "meg": resolve_path(args.meg_data_path),
        "fmri": resolve_path(args.fmri_data_path),
    }
    image_dirs = {
        "eeg": resolve_path(args.eeg_image_dir),
        "meg": resolve_path(args.meg_image_dir),
        "fmri": resolve_path(args.fmri_image_dir),
    }
    configure_dataset_image_dirs(image_dirs)
    single_checkpoints = {
        "eeg": resolve_path(args.eeg_checkpoint),
        "meg": resolve_path(args.meg_checkpoint),
        "fmri": resolve_path(args.fmri_checkpoint),
    }
    unified_checkpoint = resolve_path(args.unified_checkpoint)
    rows: list[dict[str, Any]] = []

    for model_kind in args.models:
        model = None
        for modality in args.modalities:
            if model_kind == "single":
                model = build_single_model(modality, device, single_checkpoints)
            elif model is None:
                model = build_unified_model(device, single_checkpoints, unified_checkpoint)

            subjects = [normalize_subject(s) for s in args.subjects] if args.subjects else SUBJECTS[modality]
            subjects = [sub for sub in subjects if sub in SUBJECTS[modality]]
            if not subjects:
                continue

            for subject in subjects:
                print(f"Evaluating model={model_kind} modality={modality} subject={subject}")
                row = evaluate_subject(
                    model,
                    model_kind,
                    modality,
                    subject,
                    device,
                    args.batch_size,
                    args.k_values,
                    data_paths,
                )
                print(row)
                rows.append(row)

            if model_kind == "single":
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    write_outputs(rows, resolve_path(args.output_dir), not args.no_paper_comparison)


if __name__ == "__main__":
    main()
