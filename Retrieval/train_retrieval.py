#!/usr/bin/env python
"""Unified single-modality retrieval training for BrainFLORA."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_PATHS = {
    "eeg": "./data/THINGS_EEG/Preprocessed_data_250Hz",
    "meg": "./data/THINGS_MEG/preprocessed_newsplit",
    "fmri": "./data/fmri_dataset/Preprocessed",
}
DEFAULT_SUBJECTS = {
    "eeg": [f"sub-{i:02d}" for i in range(1, 11)],
    "meg": [f"sub-{i:02d}" for i in range(1, 5)],
    "fmri": [f"sub-{i:02d}" for i in range(1, 4)],
}

EEGDataset = None
MEGDataset = None
fMRIDataset = None
eeg_encoder = None
meg_encoder = None
fmri_encoder = None
ClipLoss = None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_number(subject: str) -> int:
    match = re.search(r"\d+", subject)
    if match is None:
        raise ValueError(f"Cannot parse subject id from {subject!r}")
    return int(match.group())


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_runtime_modules() -> None:
    global EEGDataset, MEGDataset, fMRIDataset, eeg_encoder, meg_encoder, fmri_encoder, ClipLoss
    if EEGDataset is not None:
        return
    from data_preparing.eegdatasets import EEGDataset as _EEGDataset
    from data_preparing.fmri_datasets_joint_subjects import fMRIDataset as _fMRIDataset
    from data_preparing.megdatasets_averaged import MEGDataset as _MEGDataset
    from model.medformer_encoders import eeg_encoder as _eeg_encoder
    from model.medformer_encoders import fmri_encoder as _fmri_encoder
    from model.medformer_encoders import meg_encoder as _meg_encoder
    from utils.losses import ClipLoss as _ClipLoss

    EEGDataset = _EEGDataset
    MEGDataset = _MEGDataset
    fMRIDataset = _fMRIDataset
    eeg_encoder = _eeg_encoder
    meg_encoder = _meg_encoder
    fmri_encoder = _fmri_encoder
    ClipLoss = _ClipLoss


def build_dataset(modality: str, data_path: Path, subjects: list[str], train: bool):
    load_runtime_modules()
    if modality == "eeg":
        return EEGDataset(str(data_path), subjects=subjects, train=train)
    if modality == "meg":
        return MEGDataset(str(data_path), subjects=subjects, train=train)
    if modality == "fmri":
        return fMRIDataset(str(data_path), subjects=subjects, train=train)
    raise ValueError(f"Unknown modality: {modality}")


def build_model(modality: str) -> nn.Module:
    load_runtime_modules()
    if modality == "eeg":
        return eeg_encoder()
    if modality == "meg":
        return meg_encoder()
    if modality == "fmri":
        return fmri_encoder()
    raise ValueError(f"Unknown modality: {modality}")


def gallery_features(dataset, modality: str, device: torch.device) -> torch.Tensor:
    features = dataset.img_features
    if modality == "meg":
        features = features[::12]
    return features.to(device).float()


def unpack_batch(batch):
    modal, data, labels, text, text_features, img, img_features, *rest = batch
    sub_ids = rest[-1] if rest else ["sub-01"] * data.shape[0]
    return data, labels, text_features, img_features, sub_ids


def train_one_epoch(
    modality: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    gallery: torch.Tensor,
    max_batches: int | None,
) -> tuple[float, float]:
    load_runtime_modules()
    model.train()
    loss_fn = ClipLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        data, labels, text_features, img_features, sub_ids = unpack_batch(batch)
        data = data.to(device).float()
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.tensor([subject_number(s) for s in sub_ids], dtype=torch.long, device=device)
        optimizer.zero_grad()
        features = model(data, subject_ids)
        logit_scale = model.logit_scale.float()
        loss = loss_fn(features, img_features, logit_scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().item())
        pred = torch.argmax(logit_scale * features @ gallery.T, dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
        processed_batches += 1
    return total_loss / max(processed_batches, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    modality: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    gallery: torch.Tensor,
    k: int,
    max_batches: int | None,
) -> tuple[float, float]:
    load_runtime_modules()
    model.eval()
    loss_fn = ClipLoss()
    all_labels = set(range(gallery.shape[0]))
    total_loss = 0.0
    total = 0
    correct = 0
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        data, labels, text_features, img_features, sub_ids = unpack_batch(batch)
        data = data.to(device).float()
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.tensor([subject_number(s) for s in sub_ids], dtype=torch.long, device=device)
        features = model(data, subject_ids)
        logit_scale = model.logit_scale.float()
        total_loss += float(loss_fn(features, img_features, logit_scale).detach().item())
        for row, label_tensor in enumerate(labels):
            label = int(label_tensor.item())
            selected = random.sample(list(all_labels - {label}), k - 1) + [label]
            logits = logit_scale * features[row] @ gallery[selected].T
            correct += int(selected[int(torch.argmax(logits).item())] == label)
            total += 1
        processed_batches += 1
    return total_loss / max(processed_batches, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BrainFLORA retrieval encoder.")
    parser.add_argument("--modality", choices=["eeg", "meg", "fmri"], required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--test-subjects", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contrast"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/retrieval_checkpoints"))
    parser.add_argument("--encoder-type", default="Medformer")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test batch limit per epoch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_path = resolve_path(args.data_path or DEFAULT_DATA_PATHS[args.modality])
    subjects = args.subjects or DEFAULT_SUBJECTS[args.modality]
    test_subjects = args.test_subjects or [subjects[0]]
    device = torch.device(args.gpu if args.device == "gpu" and torch.cuda.is_available() else "cpu")

    train_dataset = build_dataset(args.modality, data_path, subjects, train=True)
    test_dataset = build_dataset(args.modality, data_path, test_subjects, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    train_gallery = gallery_features(train_dataset, args.modality, device)
    test_gallery = gallery_features(test_dataset, args.modality, device)
    model = build_model(args.modality).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    run_name = dt.datetime.now().strftime("%m-%d_%H-%M")
    output_dir = resolve_path(args.output_dir) / args.modality / run_name
    checkpoint_dir = resolve_path(args.checkpoint_dir) / args.modality / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    main_k = 100 if args.modality == "fmri" else 200

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            args.modality,
            model,
            train_loader,
            optimizer,
            device,
            train_gallery,
            args.max_batches,
        )
        test_loss, test_acc = evaluate(
            args.modality,
            model,
            test_loader,
            device,
            test_gallery,
            main_k,
            args.max_batches,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        results.append(row)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"{epoch + 1}.pth")

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
