"""Generate caption Prior embeddings from woPrior embeddings.

This is stage 2 of the BrainFLORA caption pipeline:

    woPrior token embeddings -> caption diffusion prior -> Prior token embeddings

The resulting tensors are consumed by ``shikra_caption.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.diffusion_prior_caption import BrainDiffusionPrior, PriorNetwork


DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "features/FLORA"
DEFAULT_PRIOR_CHECKPOINT = PROJECT_ROOT / "checkpoints/caption_checkpoints/prior_diffusion/100.pth"

SUBJECTS = {
    "eeg": [f"{i:02d}" for i in range(1, 11)],
    "meg": [f"{i:02d}" for i in range(1, 5)],
    "fmri": [f"{i:02d}" for i in range(1, 4)],
}

FEATURE_MODALITY_DIRS = {
    "eeg": "EEG",
    "meg": "MEG",
    "fmri": "fMRI",
}


def normalize_subject(subject: str) -> str:
    subject = subject.removeprefix("sub-").removeprefix("sub")
    if not subject.isdigit():
        raise ValueError(f"Invalid subject id: {subject!r}")
    return f"{int(subject):02d}"


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def torch_load(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def feature_path(root: Path, modality: str, subject: str, condition: str) -> Path:
    return (
        root
        / FEATURE_MODALITY_DIRS[modality]
        / "256_1024"
        / f"FLORA_neural_features_sub-{subject}_{condition}_test.pt"
    )


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
    if not checkpoint.exists():
        raise FileNotFoundError(f"Caption prior checkpoint does not exist: {checkpoint}")

    state = torch_load(checkpoint, map_location=device)
    config = infer_prior_config(state)
    clip_emb_dim = config["clip_emb_dim"]
    clip_seq_dim = config["clip_seq_dim"]
    depth = config["depth"]
    dim_head = config["dim_head"]
    heads = config["heads"]
    if heads * dim_head != config["inner_dim"]:
        raise ValueError(
            "Could not infer a valid caption prior attention config from "
            f"{checkpoint}: heads={heads}, dim_head={dim_head}, inner_dim={config['inner_dim']}"
        )
    timesteps = 100
    print(
        "Caption prior config: "
        f"dim={clip_emb_dim}, tokens={clip_seq_dim}, depth={depth}, heads={heads}, dim_head={dim_head}"
    )
    prior_network = PriorNetwork(
        dim=clip_emb_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb",
    )
    prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_emb_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
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
    parser = argparse.ArgumentParser(description="Generate caption Prior embeddings.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modalities", nargs="+", choices=["eeg", "meg", "fmri"], default=["eeg", "meg", "fmri"])
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset, e.g. 01 sub-02")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument("--sampling-timesteps", type=int, default=20)
    parser.add_argument("--cond-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit per subject.")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs/caption_embedding/caption_prior_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    features_root = resolve_path(args.features_root)
    prior = build_caption_prior(device, resolve_path(args.prior_checkpoint))
    summary = []

    for modality in args.modalities:
        subjects = [normalize_subject(s) for s in args.subjects] if args.subjects else SUBJECTS[modality]
        subjects = [sub for sub in subjects if sub in SUBJECTS[modality]]
        for subject in subjects:
            input_path = feature_path(features_root, modality, subject, "woPrior")
            output_path = feature_path(features_root, modality, subject, "Prior")
            if not input_path.exists():
                raise FileNotFoundError(f"woPrior embedding does not exist: {input_path}")
            wo_prior = torch_load(input_path, map_location="cpu")
            if not torch.is_tensor(wo_prior):
                raise TypeError(f"Expected tensor in {input_path}, got {type(wo_prior)!r}")
            if args.max_samples is not None:
                wo_prior = wo_prior[: args.max_samples]
            print(f"Generating Prior embeddings for {modality} sub-{subject}: {tuple(wo_prior.shape)}")
            prior_embeddings = generate_prior_embeddings(
                prior,
                wo_prior,
                device,
                args.sampling_timesteps,
                args.cond_scale,
                args.batch_size,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prior_embeddings, output_path)
            summary.append(
                {
                    "modality": modality,
                    "subject": f"sub-{subject}",
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


if __name__ == "__main__":
    main()
