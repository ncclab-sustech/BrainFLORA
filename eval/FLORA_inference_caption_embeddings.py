"""Generate BrainFLORA caption embeddings with optional diffusion prior.

Stages:
  - woPrior: brain signal -> caption UnifiedEncoder -> token embeddings
  - Prior: woPrior token embeddings -> caption diffusion prior -> prior embeddings
  - all: run woPrior and Prior in sequence
"""

from __future__ import annotations

import argparse
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
DEFAULT_CAPTION_CHECKPOINT = PROJECT_ROOT / "checkpoints/caption_checkpoints/90.pth"
DEFAULT_PRIOR_CHECKPOINT = PROJECT_ROOT / "checkpoints/caption_checkpoints/prior_diffusion/100.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "features/FLORA"

SUBJECTS = {
    "eeg": [f"sub-{i:02d}" for i in range(1, 11)],
    "meg": [f"sub-{i:02d}" for i in range(1, 5)],
    "fmri": [f"sub-{i:02d}" for i in range(1, 4)],
}
FEATURE_MODALITY_DIRS = {"eeg": "EEG", "meg": "MEG", "fmri": "fMRI"}

eeg_dataset_module = None
meg_dataset_module = None
fmri_dataset_module = None
EEGDataset = None
MEGDataset = None
fMRIDataset = None
BrainDiffusionPrior = None
PriorNetwork = None
UnifiedEncoder = None
ClipLoss = None


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


def load_runtime_modules(stage: str = "all") -> None:
    global eeg_dataset_module, meg_dataset_module, fmri_dataset_module
    global EEGDataset, MEGDataset, fMRIDataset, BrainDiffusionPrior, PriorNetwork, UnifiedEncoder, ClipLoss
    if stage in {"all", "woPrior"} and EEGDataset is None:
        from data_preparing import eegdatasets as _eeg_dataset_module
        from data_preparing import fmri_datasets_joint_subjects as _fmri_dataset_module
        from data_preparing import megdatasets_averaged as _meg_dataset_module
        from data_preparing.eegdatasets import EEGDataset as _EEGDataset
        from data_preparing.fmri_datasets_joint_subjects import fMRIDataset as _fMRIDataset
        from data_preparing.megdatasets_averaged import MEGDataset as _MEGDataset
        from model.unified_encoder_multi_tower import UnifiedEncoder as _UnifiedEncoder
        from utils.losses import ClipLoss as _ClipLoss

        eeg_dataset_module = _eeg_dataset_module
        meg_dataset_module = _meg_dataset_module
        fmri_dataset_module = _fmri_dataset_module
        EEGDataset = _EEGDataset
        MEGDataset = _MEGDataset
        fMRIDataset = _fMRIDataset
        UnifiedEncoder = _UnifiedEncoder
        ClipLoss = _ClipLoss
    if stage in {"all", "Prior"} and BrainDiffusionPrior is None:
        from model.diffusion_prior_caption import BrainDiffusionPrior as _BrainDiffusionPrior
        from model.diffusion_prior_caption import PriorNetwork as _PriorNetwork

        BrainDiffusionPrior = _BrainDiffusionPrior
        PriorNetwork = _PriorNetwork


def configure_dataset_image_dirs(image_dirs: dict[str, Path]) -> None:
    load_runtime_modules("woPrior")
    eeg_dataset_module.img_directory_test = str(image_dirs["eeg"])
    meg_dataset_module.img_directory_test = str(image_dirs["meg"])
    fmri_dataset_module.img_directory_test = str(image_dirs["fmri"])


def build_dataset(modality: str, subject: str, data_paths: dict[str, Path], use_caption_features: bool):
    load_runtime_modules("woPrior")
    if modality == "eeg":
        return EEGDataset(str(data_paths["eeg"]), subjects=[subject], train=False, use_caption=use_caption_features)
    if modality == "meg":
        return MEGDataset(str(data_paths["meg"]), subjects=[subject], train=False, use_caption=use_caption_features)
    if modality == "fmri":
        return fMRIDataset(
            str(data_paths["fmri"]),
            adap_subject=subject,
            subjects=[subject],
            train=False,
            use_caption=use_caption_features,
        )
    raise ValueError(f"Unknown modality: {modality}")


def candidate_img_features(dataset, modality: str) -> torch.Tensor:
    return dataset.img_features[::12] if modality == "meg" else dataset.img_features


def feature_path(root: Path, modality: str, subject: str, condition: str) -> Path:
    subject_id = f"{subject_number(subject):02d}"
    return (
        root
        / FEATURE_MODALITY_DIRS[modality]
        / "256_1024"
        / f"FLORA_neural_features_sub-{subject_id}_{condition}_test.pt"
    )


def build_caption_model(
    device: torch.device,
    single_checkpoints: dict[str, Path],
    caption_checkpoint: Path,
) -> UnifiedEncoder:
    load_runtime_modules("woPrior")
    for modality, checkpoint in single_checkpoints.items():
        require_path(checkpoint, f"{modality} checkpoint")
    require_path(caption_checkpoint, "caption unified checkpoint")
    model = UnifiedEncoder(
        {modality: str(path) for modality, path in single_checkpoints.items()},
        device=device,
        user_caption=True,
    )
    model.load_state_dict(torch_load(caption_checkpoint, map_location=device), strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_subject_embeddings(
    model: UnifiedEncoder,
    modality: str,
    subject: str,
    data_paths: dict[str, Path],
    device: torch.device,
    batch_size: int,
    k_values: list[int],
    use_caption_features: bool,
    max_batches: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    load_runtime_modules("woPrior")
    dataset = build_dataset(modality, subject, data_paths, use_caption_features)
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
    token_embeddings = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        _, data, labels, _, _, _, img_features, _, _, sub_ids = batch
        data = data.to(device).float()
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.tensor(
            [subject_number(sub_id) for sub_id in sub_ids],
            dtype=torch.long,
            device=device,
        )
        ret_emb, neural_tokens = model(data, subject_ids, modal=modality)
        token_embeddings.append(neural_tokens.cpu())
        total_loss += loss_func(ret_emb.float(), img_features.float(), logit_scale).item()
        total_batches += 1
        total_samples += labels.numel()

        for row, label_tensor in enumerate(labels):
            label = int(label_tensor.item())
            possible = list(all_labels - {label})
            for k in valid_k_values:
                selected = random.sample(possible, k - 1) + [label]
                logits = logit_scale * ret_emb[row].float() @ img_features_all[selected].T
                predicted = selected[int(torch.argmax(logits).item())]
                totals[k] += 1
                correct[k] += int(predicted == label)
                if k >= 5:
                    _, top_indices = torch.topk(logits, 5, largest=True)
                    top5_correct[k] += int(label in [selected[i] for i in top_indices.tolist()])

    metrics: dict[str, Any] = {
        "modality": modality,
        "subject": subject,
        "n_classes": n_classes,
        "n_samples": total_samples,
        "loss": total_loss / max(total_batches, 1),
    }
    for k in valid_k_values:
        metrics[f"acc@{k}"] = correct[k] / max(totals[k], 1)
        if k >= 5:
            metrics[f"top5@{k}"] = top5_correct[k] / max(totals[k], 1)
    return torch.cat(token_embeddings, dim=0), metrics


def infer_prior_config(state: dict[str, torch.Tensor]) -> dict[str, int]:
    null_image_embed = state["net.null_image_embed"]
    to_q = state["net.causal_transformer.layers.0.0.to_q.weight"]
    null_kv = state["net.causal_transformer.layers.0.0.null_kv"]
    rel_pos = state["net.causal_transformer.rel_pos_bias.relative_attention_bias.weight"]
    depth = len(
        {
            int(key.split(".")[3])
            for key in state
            if key.startswith("net.causal_transformer.layers.")
        }
    )
    return {
        "clip_seq_dim": int(null_image_embed.shape[0]),
        "clip_emb_dim": int(null_image_embed.shape[1]),
        "depth": depth,
        "dim_head": int(null_kv.shape[1]),
        "heads": int(rel_pos.shape[1]),
        "inner_dim": int(to_q.shape[0]),
    }


def build_caption_prior(device: torch.device, checkpoint: Path) -> BrainDiffusionPrior:
    load_runtime_modules("Prior")
    require_path(checkpoint, "caption prior checkpoint")
    state = torch_load(checkpoint, map_location=device)
    config = infer_prior_config(state)
    if config["heads"] * config["dim_head"] != config["inner_dim"]:
        raise ValueError(
            "Invalid caption prior config inferred from checkpoint: "
            f"heads={config['heads']}, dim_head={config['dim_head']}, inner_dim={config['inner_dim']}"
        )
    prior_network = PriorNetwork(
        dim=config["clip_emb_dim"],
        depth=config["depth"],
        dim_head=config["dim_head"],
        heads=config["heads"],
        causal=False,
        num_tokens=config["clip_seq_dim"],
        learned_query_mode="pos_emb",
    )
    prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=config["clip_emb_dim"],
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    prior.load_state_dict(state, strict=True)
    prior.to(device)
    prior.eval()
    return prior


def generate_prior_embeddings(
    prior: BrainDiffusionPrior,
    wo_prior_embeddings: torch.Tensor,
    device: torch.device,
    sampling_timesteps: int,
    cond_scale: float,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, wo_prior_embeddings.shape[0], batch_size):
            batch = wo_prior_embeddings[start : start + batch_size].to(device).float()
            prior_out = prior.p_sample_loop(
                batch.shape,
                text_cond={"text_embed": batch},
                cond_scale=cond_scale,
                timesteps=sampling_timesteps,
            )
            outputs.append(prior_out.detach().cpu())
            print(f"Generated {min(start + batch_size, wo_prior_embeddings.shape[0])}/{wo_prior_embeddings.shape[0]}")
    return torch.cat(outputs, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BrainFLORA caption embeddings.")
    parser.add_argument("--stage", choices=["woPrior", "Prior", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prior-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="+", choices=["eeg", "meg", "fmri"], default=["eeg", "meg", "fmri"])
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset, e.g. sub-01 02")
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 10, 50, 100, 200])
    parser.add_argument("--features-root", "--output-root", dest="features_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-json", type=Path, default=PROJECT_ROOT / "outputs/caption_embedding/caption_woPrior_metrics.json")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs/caption_embedding/caption_prior_summary.json")
    parser.add_argument("--use-caption-features", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional woPrior smoke-test limit per subject.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional Prior smoke-test limit per subject.")
    parser.add_argument("--sampling-timesteps", type=int, default=20)
    parser.add_argument("--cond-scale", type=float, default=1.0)
    parser.add_argument("--eeg-data-path", type=Path, default=DEFAULT_DATA_PATHS["eeg"])
    parser.add_argument("--meg-data-path", type=Path, default=DEFAULT_DATA_PATHS["meg"])
    parser.add_argument("--fmri-data-path", type=Path, default=DEFAULT_DATA_PATHS["fmri"])
    parser.add_argument("--eeg-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["eeg"])
    parser.add_argument("--meg-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["meg"])
    parser.add_argument("--fmri-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["fmri"])
    parser.add_argument("--eeg-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["eeg"])
    parser.add_argument("--meg-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["meg"])
    parser.add_argument("--fmri-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["fmri"])
    parser.add_argument("--caption-checkpoint", type=Path, default=DEFAULT_CAPTION_CHECKPOINT)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    return parser.parse_args()


def selected_subjects(modality: str, subjects: list[str] | None) -> list[str]:
    values = [normalize_subject(s) for s in subjects] if subjects else SUBJECTS[modality]
    return [subject for subject in values if subject in SUBJECTS[modality]]


def run_wo_prior(args: argparse.Namespace, device: torch.device, features_root: Path) -> list[dict[str, Any]]:
    data_paths = {
        "eeg": resolve_path(args.eeg_data_path),
        "meg": resolve_path(args.meg_data_path),
        "fmri": resolve_path(args.fmri_data_path),
    }
    configure_dataset_image_dirs(
        {
            "eeg": resolve_path(args.eeg_image_dir),
            "meg": resolve_path(args.meg_image_dir),
            "fmri": resolve_path(args.fmri_image_dir),
        }
    )
    model = build_caption_model(
        device,
        {
            "eeg": resolve_path(args.eeg_checkpoint),
            "meg": resolve_path(args.meg_checkpoint),
            "fmri": resolve_path(args.fmri_checkpoint),
        },
        resolve_path(args.caption_checkpoint),
    )
    metrics = []
    for modality in args.modalities:
        for subject in selected_subjects(modality, args.subjects):
            print(f"Generating woPrior embeddings for modality={modality} subject={subject}")
            embeddings, row = extract_subject_embeddings(
                model,
                modality,
                subject,
                data_paths,
                device,
                args.batch_size,
                args.k_values,
                args.use_caption_features,
                args.max_batches,
            )
            path = feature_path(features_root, modality, subject, "woPrior")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, path)
            row["embedding_path"] = str(path)
            row["embedding_shape"] = list(embeddings.shape)
            metrics.append(row)
            print(f"Wrote {path} with shape {tuple(embeddings.shape)}")
    metrics_path = resolve_path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {metrics_path}")
    return metrics


def run_prior(args: argparse.Namespace, device: torch.device, features_root: Path) -> list[dict[str, Any]]:
    prior = build_caption_prior(device, resolve_path(args.prior_checkpoint))
    summary = []
    for modality in args.modalities:
        for subject in selected_subjects(modality, args.subjects):
            input_path = feature_path(features_root, modality, subject, "woPrior")
            output_path = feature_path(features_root, modality, subject, "Prior")
            require_path(input_path, "woPrior embedding")
            wo_prior = torch_load(input_path, map_location="cpu")
            if not torch.is_tensor(wo_prior):
                raise TypeError(f"Expected tensor in {input_path}, got {type(wo_prior)!r}")
            if args.max_samples is not None:
                wo_prior = wo_prior[: args.max_samples]
            print(f"Generating Prior embeddings for {modality} {subject}: {tuple(wo_prior.shape)}")
            prior_embeddings = generate_prior_embeddings(
                prior,
                wo_prior,
                device,
                args.sampling_timesteps,
                args.cond_scale,
                args.prior_batch_size,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prior_embeddings, output_path)
            summary.append(
                {
                    "modality": modality,
                    "subject": subject,
                    "input": str(input_path),
                    "output": str(output_path),
                    "shape": list(prior_embeddings.shape),
                }
            )
            print(f"Wrote {output_path}")
    summary_path = resolve_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    features_root = resolve_path(args.features_root)
    if args.stage in {"woPrior", "all"}:
        run_wo_prior(args, device, features_root)
    if args.stage in {"Prior", "all"}:
        run_prior(args, device, features_root)


if __name__ == "__main__":
    main()
