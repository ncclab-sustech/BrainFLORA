#!/usr/bin/env python
"""Unified BrainFLORA encoder training entry point.

This script replaces the historical split training files with one CLI:

  - retrieval: train the unified neural encoder for image retrieval.
  - reconstruction: train retrieval plus the 1024-d high-level diffusion prior.
  - caption: train caption-aligned CLIP patch-token embeddings for Shikra captions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import warnings
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

warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")

MODALITIES = ("eeg", "meg", "fmri")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def extract_id_from_string(value: str) -> int:
    match = re.search(r"\d+", value)
    if match is None:
        raise ValueError(f"Could not extract numeric subject id from {value!r}")
    return int(match.group())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_encoder_paths(values: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for value in values:
        key, sep, path = value.partition("=")
        if sep != "=" or key not in MODALITIES:
            raise ValueError(
                "Encoder paths must use '<modality>=<path>', e.g. eeg=checkpoints/eeg_encoder.pth"
            )
        paths[key] = path
    return paths


def has_option(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def maybe_add_default(argv: list[str], flag: str, value: str) -> list[str]:
    if has_option(argv, flag):
        return argv
    return [*argv, flag, value]


def get_device(args: argparse.Namespace, accelerator: Any | None = None) -> torch.device:
    if accelerator is not None:
        return accelerator.device
    if args.device == "gpu" and torch.cuda.is_available():
        return torch.device(args.gpu)
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BrainFLORA unified neural encoders")
    parser.add_argument(
        "--task",
        choices=["retrieval", "reconstruction", "caption"],
        default="retrieval",
        help="Training objective to run.",
    )
    parser.add_argument(
        "--encoder_paths",
        nargs="+",
        default=[
            "eeg=./checkpoints/eeg_encoder.pth",
            "meg=./checkpoints/meg_encoder.pth",
            "fmri=./checkpoints/fmri_encoder.pth",
        ],
        help="Pretrained single-modality encoder paths as '<modality>=<path>'.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        default=list(MODALITIES),
        help="Modalities used for unified training.",
    )
    parser.add_argument(
        "--eval_modality",
        choices=MODALITIES,
        default="fmri",
        help="Modality used for validation during training.",
    )
    parser.add_argument("--eeg_data_path", default="./data/THINGS_EEG/Preprocessed_data_250Hz")
    parser.add_argument("--meg_data_path", default="./data/THINGS_MEG/preprocessed_newsplit")
    parser.add_argument("--fmri_data_path", default="./data/fmri_dataset/Preprocessed")
    parser.add_argument("--output_dir", default="./outputs/contrast")
    parser.add_argument("--project", default="train_pos_img_text_rep")
    parser.add_argument("--entity", default="sustech_rethinkingbci")
    parser.add_argument("--name", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prior_lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--insubject", nargs="?", const=True, default=True, type=str_to_bool)
    parser.add_argument("--encoder_type", default="Unified_EEG+MEG+fMRI_EEG")
    parser.add_argument("--test_subjects", nargs="+", default=["sub-02"])
    parser.add_argument(
        "--eeg_subjects",
        nargs="+",
        default=[
            "sub-01",
            "sub-02",
            "sub-03",
            "sub-04",
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-08",
            "sub-09",
            "sub-10",
        ],
    )
    parser.add_argument("--meg_subjects", nargs="+", default=["sub-01", "sub-02", "sub-03", "sub-04"])
    parser.add_argument("--fmri_subjects", nargs="+", default=["sub-01", "sub-02", "sub-03"])
    parser.add_argument(
        "--use-prior",
        "--use_prior",
        dest="use_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the task-specific diffusion prior when supported.",
    )
    parser.add_argument(
        "--use-caption",
        "--use_caption",
        dest="use_caption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load caption CLIP token features. Defaults to true for --task caption only.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use accelerate for distributed/mixed-precision training.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Accelerate mixed precision mode.",
    )
    return parser


def load_training_datasets(args: argparse.Namespace, use_caption: bool) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from data_preparing.datasets_mixer import (
        MetaDataLoader,
        MetaEEGDataset,
        MetaMEGDataset,
        MetafMRIDataset,
    )

    datasets: dict[str, Any] = {"eeg": None, "meg": None, "fmri": None}
    text_features: dict[str, Any] = {}
    img_features: dict[str, Any] = {}

    if "eeg" in args.modalities:
        datasets["eeg"] = MetaEEGDataset(
            args.eeg_data_path, args.eeg_subjects, train=True, use_caption=use_caption
        )
        text_features["eeg"] = datasets["eeg"].text_features
        img_features["eeg"] = datasets["eeg"].img_features

    if "meg" in args.modalities:
        datasets["meg"] = MetaMEGDataset(
            args.meg_data_path, args.meg_subjects, train=True, use_caption=use_caption
        )
        text_features["meg"] = datasets["meg"].text_features
        img_features["meg"] = datasets["meg"].img_features

    if "fmri" in args.modalities:
        datasets["fmri"] = MetafMRIDataset(
            args.fmri_data_path, args.fmri_subjects, train=True, use_caption=use_caption
        )
        text_features["fmri"] = datasets["fmri"].text_features
        img_features["fmri"] = datasets["fmri"].img_features

    train_loader = MetaDataLoader(
        eeg_dataset=datasets["eeg"],
        meg_dataset=datasets["meg"],
        fmri_dataset=datasets["fmri"],
        batch_size=args.batch_size,
        drop_last=True,
        modalities=args.modalities,
    )
    return train_loader, text_features, img_features


def load_eval_dataset(args: argparse.Namespace, use_caption: bool) -> tuple[DataLoader, dict[str, Any], dict[str, Any]]:
    from data_preparing.eegdatasets import EEGDataset
    from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
    from data_preparing.megdatasets_averaged import MEGDataset

    if args.eval_modality == "eeg":
        dataset = EEGDataset(args.eeg_data_path, subjects=args.test_subjects, train=False, use_caption=use_caption)
    elif args.eval_modality == "meg":
        dataset = MEGDataset(args.meg_data_path, subjects=args.test_subjects, train=False, use_caption=use_caption)
    else:
        dataset = fMRIDataset(args.fmri_data_path, subjects=args.test_subjects, train=False, use_caption=use_caption)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    return loader, {args.eval_modality: dataset.text_features}, {args.eval_modality: dataset.img_features}


def pool_image_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 3:
        return features.mean(dim=1)
    return features


def select_train_gallery(features: dict[str, Any], modality: str, device: torch.device, use_caption: bool) -> torch.Tensor:
    selected = features[modality]
    if not use_caption:
        selected = selected[::10] if modality == "eeg" else selected[::12]
    selected = pool_image_features(selected).to(device).float()
    return selected


def select_eval_gallery(features: dict[str, Any], modality: str, device: torch.device, use_caption: bool) -> torch.Tensor:
    selected = features[modality]
    if not use_caption and modality == "meg":
        selected = selected[::12]
    selected = pool_image_features(selected).to(device).float()
    return selected


def forward_unified(
    model: Any, data: torch.Tensor, subject_ids: torch.Tensor, modal: str, task: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(data, subject_ids, modal=modal)
    if task == "caption":
        neural_features, token_features = output
        return neural_features, token_features
    return output, None


def build_reconstruction_prior(device: torch.device) -> Any:
    from model.diffusion_prior import DiffusionPriorUNet, Pipe

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    return Pipe(diffusion_prior, device=device)


def build_caption_prior(device: torch.device) -> Any:
    from model.diffusion_prior_caption import BrainDiffusionPrior, Pipe, PriorNetwork

    clip_emb_dim = 1024
    clip_seq_dim = 256
    dim_head = 52
    prior_network = PriorNetwork(
        dim=clip_emb_dim,
        depth=6,
        dim_head=dim_head,
        heads=clip_emb_dim // dim_head,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb",
    )
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_emb_dim,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    return Pipe(diffusion_prior, device=device)


def create_optimizer(args: argparse.Namespace, model: nn.Module, high_pipe: Any | None) -> AdamW:
    parameter_groups: list[dict[str, Any]] = [{"params": model.parameters(), "lr": args.lr}]
    if high_pipe is not None and args.use_prior:
        parameter_groups.append({"params": high_pipe.diffusion_prior.parameters(), "lr": args.prior_lr})
    return AdamW(parameter_groups)


def maybe_create_scheduler(args: argparse.Namespace, optimizer: AdamW, high_pipe: Any | None, train_loader: Any) -> Any | None:
    if high_pipe is None or not args.use_prior or args.task != "reconstruction":
        return None
    from diffusers.optimization import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * args.epochs,
    )
    high_pipe.diffusion_prior.lr_scheduler = scheduler
    return scheduler


def compute_prior_loss(
    args: argparse.Namespace,
    epoch: int,
    high_pipe: Any | None,
    neural_features: torch.Tensor,
    token_features: torch.Tensor | None,
    img_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if high_pipe is None or not args.use_prior or epoch <= int(0.33 * args.epochs):
        return torch.zeros((), device=device)

    if args.task == "caption":
        if token_features is None:
            raise RuntimeError("Caption training requires token features from UnifiedEncoder")
        prior_loss, _ = high_pipe.diffusion_prior(text_embed=token_features, image_embed=img_features)
        return prior_loss

    num_train_timesteps = high_pipe.scheduler.config.num_train_timesteps
    c_embeds = neural_features
    h_embeds = img_features
    noise = torch.randn_like(h_embeds)
    timesteps = torch.randint(0, num_train_timesteps, (h_embeds.shape[0],), device=device)
    if torch.rand(1, device=device) < 0.1:
        c_embeds = None
    perturbed_h_embeds = high_pipe.scheduler.add_noise(h_embeds, noise, timesteps)
    noise_pred = high_pipe.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
    return nn.functional.mse_loss(noise_pred, noise) * 30.0


def train_one_epoch(
    epoch: int,
    args: argparse.Namespace,
    model: Any,
    high_pipe: Any | None,
    dataloader: Any,
    optimizer: AdamW,
    lr_scheduler: Any | None,
    device: torch.device,
    img_features_all: dict[str, Any],
    accelerator: Any | None = None,
) -> tuple[float, float]:
    from utils.losses import ClipLoss, mixco_1d, mixco_nce, soft_clip_loss

    model.train()
    if high_pipe is not None:
        high_pipe.diffusion_prior.train()

    total_loss = 0.0
    correct = 0
    total = 0
    clip_loss_fn = ClipLoss()
    mse_loss_fn = nn.MSELoss(reduction="mean")
    gallery = select_train_gallery(img_features_all, args.eval_modality, device, args.use_caption)

    for batch_idx, batch in enumerate(dataloader):
        modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids = batch
        data = data.to(device).float()
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.tensor([extract_id_from_string(sub_id) for sub_id in sub_ids], dtype=torch.long, device=device)

        optimizer.zero_grad()
        neural_features, token_features = forward_unified(model, data, subject_ids, modal[0], args.task)
        logit_scale = model.logit_scale.float()

        if args.task == "retrieval":
            loss = clip_loss_fn(neural_features, pool_image_features(img_features), logit_scale)
        else:
            pooled_img_features = pool_image_features(img_features)
            regress_target = img_features if args.task == "caption" else pooled_img_features
            regress_source = token_features if args.task == "caption" else neural_features
            regress_loss = mse_loss_fn(regress_source, regress_target)
            if args.task == "reconstruction":
                regress_loss = regress_loss * 30.0

            neural_for_clip, perm, betas, select = mixco_1d(neural_features.clone())
            neural_for_clip = nn.functional.normalize(neural_for_clip.flatten(1), dim=-1)
            img_for_clip = nn.functional.normalize(pooled_img_features.flatten(1), dim=-1)
            if epoch < int(0.1 * args.epochs):
                loss_clip = mixco_nce(
                    neural_for_clip,
                    img_for_clip,
                    temp=0.006,
                    perm=perm,
                    betas=betas,
                    select=select,
                )
            else:
                loss_clip = soft_clip_loss(neural_for_clip, img_for_clip, temp=logit_scale)

            prior_loss = compute_prior_loss(
                args, epoch, high_pipe, neural_features, token_features, img_features, device
            )
            loss = loss_clip + regress_loss + prior_loss

        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if high_pipe is not None and args.use_prior:
            torch.nn.utils.clip_grad_norm_(high_pipe.diffusion_prior.parameters(), max_norm=1.0)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += float(loss.detach().item())
        logits_img = logit_scale * neural_features @ gallery.T
        predicted = torch.argmax(logits_img, dim=1)
        total += predicted.shape[0]
        correct += (predicted == labels).sum().item()

        del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids

    return total_loss / (batch_idx + 1), correct / max(total, 1)


def evaluate(
    epoch: int,
    args: argparse.Namespace,
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    img_features_all: dict[str, Any],
    k: int,
) -> tuple[float, float, float]:
    from utils.losses import mixco_1d, mixco_nce, soft_clip_loss

    model.eval()
    gallery = select_eval_gallery(img_features_all, args.eval_modality, device, args.use_caption)
    all_labels = set(range(gallery.shape[0]))
    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0
    batch_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids = batch
            data = data.to(device).float()
            labels = labels.to(device)
            img_features = pool_image_features(img_features.to(device).float())
            subject_ids = torch.tensor(
                [extract_id_from_string(sub_id) for sub_id in sub_ids], dtype=torch.long, device=device
            )

            neural_features, _ = forward_unified(model, data, subject_ids, args.eval_modality, args.task)
            logit_scale = model.logit_scale.float()
            neural_for_clip, perm, betas, select = mixco_1d(neural_features)
            neural_for_clip = nn.functional.normalize(neural_for_clip.flatten(1), dim=-1)
            img_for_clip = nn.functional.normalize(img_features.flatten(1), dim=-1)
            if epoch < int(0.33 * args.epochs):
                loss_clip = mixco_nce(
                    neural_for_clip,
                    img_for_clip,
                    temp=0.006,
                    perm=perm,
                    betas=betas,
                    select=select,
                )
            else:
                loss_clip = soft_clip_loss(neural_for_clip, img_for_clip, temp=logit_scale)
            total_loss += float(loss_clip.detach().item())

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                selected_img_features = gallery[selected_classes]
                logits = logit_scale * neural_features[idx] @ selected_img_features.T
                predicted_label = selected_classes[torch.argmax(logits).item()]
                correct += int(predicted_label == label.item())
                total += 1
                if k >= 5:
                    _, top5_indices = torch.topk(logits, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct += 1

            del modal, data, labels, text, text_features, img, img_features, index, img_index, sub_ids

    return total_loss / (batch_idx + 1), correct / max(total, 1), top5_correct / max(total, 1)


def checkpoint_state_dict(model: nn.Module, accelerator: Any | None) -> dict[str, torch.Tensor]:
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    return model.state_dict()


def save_checkpoints(
    args: argparse.Namespace,
    model: Any,
    high_pipe: Any | None,
    epoch: int,
    run_dir: Path,
    accelerator: Any | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state_dict(model, accelerator), run_dir / f"{epoch}.pth")
    if high_pipe is not None and args.use_prior:
        prior_dir = run_dir / "prior_diffusion"
        prior_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_state_dict(high_pipe.diffusion_prior, accelerator), prior_dir / f"{epoch}.pth")


def train(args: argparse.Namespace) -> list[dict[str, float]]:
    from model.unified_encoder_multi_tower import UnifiedEncoder

    set_seed(args.seed)
    args.use_caption = args.task == "caption" if args.use_caption is None else args.use_caption
    if args.task == "caption" and not args.use_caption:
        raise ValueError("--task caption requires caption features; remove --no-use-caption")

    accelerator = None
    if args.distributed:
        from accelerate import Accelerator

        accelerator = Accelerator(
            device_placement=True,
            split_batches=True,
            mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        )
        print_fn = accelerator.print
    else:
        print_fn = print

    device = get_device(args, accelerator)
    encoder_paths = parse_encoder_paths(args.encoder_paths)
    run_name = args.name or f"{args.task}_{dt.datetime.now().strftime('%m-%d_%H-%M')}"
    args.name = run_name

    train_loader, text_features_train_all, img_features_train_all = load_training_datasets(args, args.use_caption)
    test_loader, text_features_test_all, img_features_test_all = load_eval_dataset(args, args.use_caption)

    model = UnifiedEncoder(
        encoder_paths,
        device,
        num_experts=5,
        num_heads=args.depth,
        ff_dim=64 * args.depth,
        num_layers=args.depth,
        user_caption=args.task == "caption",
    )
    model.to(device)

    high_pipe = None
    if args.use_prior and args.task == "reconstruction":
        high_pipe = build_reconstruction_prior(device)
    elif args.use_prior and args.task == "caption":
        high_pipe = build_caption_prior(device)

    optimizer = create_optimizer(args, model, high_pipe)

    if accelerator is not None:
        if high_pipe is None:
            model, optimizer, train_loader, test_loader = accelerator.prepare(
                model, optimizer, train_loader, test_loader
            )
        else:
            model, high_pipe.diffusion_prior, optimizer, train_loader, test_loader = accelerator.prepare(
                model, high_pipe.diffusion_prior, optimizer, train_loader, test_loader
            )

    lr_scheduler = maybe_create_scheduler(args, optimizer, high_pipe, train_loader)
    logger = None
    if args.logger:
        from utils import wandb_logger

        logger = wandb_logger(args)
    if logger is not None:
        logger.watch(model, "gradients")

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print_fn(f"Task: {args.task}")
    print_fn(f"Device: {device}")
    print_fn(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    run_dir = Path(args.output_dir) / args.task / args.encoder_type / run_name
    results: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            epoch,
            args,
            model,
            high_pipe,
            train_loader,
            optimizer,
            lr_scheduler,
            device,
            img_features_train_all,
            accelerator,
        )
        main_k = 100 if args.eval_modality == "fmri" else 200
        test_loss, test_acc, top5_acc = evaluate(
            epoch, args, model, test_loader, device, img_features_test_all, k=main_k
        )
        _, v2_acc, _ = evaluate(epoch, args, model, test_loader, device, img_features_test_all, k=2)
        _, v4_acc, _ = evaluate(epoch, args, model, test_loader, device, img_features_test_all, k=4)
        _, v10_acc, _ = evaluate(epoch, args, model, test_loader, device, img_features_test_all, k=10)
        _, v50_acc, v50_top5_acc = evaluate(epoch, args, model, test_loader, device, img_features_test_all, k=50)
        _, v100_acc, v100_top5_acc = evaluate(epoch, args, model, test_loader, device, img_features_test_all, k=100)

        epoch_result = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "top5_acc": float(top5_acc),
            "v2_acc": float(v2_acc),
            "v4_acc": float(v4_acc),
            "v10_acc": float(v10_acc),
            "v50_acc": float(v50_acc),
            "v100_acc": float(v100_acc),
            "v50_top5_acc": float(v50_top5_acc),
            "v100_top5_acc": float(v100_top5_acc),
        }
        results.append(epoch_result)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1

        if logger is not None:
            logger.log(epoch_result, step=epoch + 1)

        print_fn(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} top5={top5_acc:.4f}"
        )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoints(args, model, high_pipe, epoch + 1, run_dir, accelerator)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "results": results}, f, indent=2)

    if logger is not None:
        logger.finish()
    print_fn(f"Best epoch: {best_epoch}, best test accuracy: {best_accuracy:.4f}")
    print_fn(f"Saved outputs to: {run_dir}")
    return results


def main(argv: list[str] | None = None) -> list[dict[str, float]]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
