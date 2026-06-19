"""Run BrainFLORA visual reconstruction from pretrained checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

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
from model.diffusion_prior import DiffusionPriorUNet, Pipe
from model.custom_pipeline import Generator4Embeds
from model.unified_encoder_multi_tower import UnifiedEncoder


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

DEFAULT_RECONSTRUCTION_CHECKPOINT = PROJECT_ROOT / "checkpoints/reconstruction_checkpoints/150.pth"
DEFAULT_PRIOR_CHECKPOINT = PROJECT_ROOT / "checkpoints/reconstruction_checkpoints/prior_diffusion/150.pth"

SUBJECTS = {
    "eeg": [f"sub-{i:02d}" for i in range(1, 11)],
    "meg": [f"sub-{i:02d}" for i in range(1, 5)],
    "fmri": [f"sub-{i:02d}" for i in range(1, 4)],
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


def build_unified_model(
    device: torch.device,
    single_checkpoints: dict[str, Path],
    reconstruction_checkpoint: Path,
    depth: int,
) -> UnifiedEncoder:
    for modality, checkpoint in single_checkpoints.items():
        require_path(checkpoint, f"{modality} checkpoint")
    require_path(reconstruction_checkpoint, "reconstruction unified checkpoint")

    model = UnifiedEncoder(
        {modality: str(path) for modality, path in single_checkpoints.items()},
        device=device,
        num_experts=5,
        num_heads=depth,
        ff_dim=64 * depth,
        num_layers=depth,
        user_caption=False,
    )
    state = torch_load(reconstruction_checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_prior_pipe(device: torch.device, prior_checkpoint: Path) -> Pipe:
    require_path(prior_checkpoint, "reconstruction prior checkpoint")
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(torch_load(prior_checkpoint, map_location=device), strict=True)
    pipe.diffusion_prior.to(device)
    pipe.diffusion_prior.eval()
    return pipe


@torch.no_grad()
def reconstruct_subject(
    *,
    model: UnifiedEncoder,
    prior_pipe: Pipe,
    generator: Optional[Generator4Embeds],
    modality: str,
    subject: str,
    data_paths: dict[str, Path],
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    prior_steps: int,
    image_guidance_scale: float,
    seed: int,
    max_images: int | None,
    print_every: int,
    skip_existing: bool,
) -> dict[str, object]:
    dataset = build_dataset(modality, subject, data_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    subject_dir = output_dir / modality / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed)
    count = 0
    skipped = 0
    generated = 0

    for batch in loader:
        if max_images is not None and count >= max_images:
            break
        _, data, _, _, _, _, _, _, _, _ = batch
        data = data.to(device).float()
        subject_ids = torch.full(
            (data.shape[0],),
            subject_number(subject),
            dtype=torch.long,
            device=device,
        )
        neural_features = model(data, subject_ids, modal=modality)
        for row in range(neural_features.shape[0]):
            if max_images is not None and count >= max_images:
                break
            count += 1
            if generator is None:
                path = subject_dir / f"high_level_{count:04d}.pt"
            else:
                path = subject_dir / f"image_{count:04d}.png"
            if skip_existing and path.exists():
                skipped += 1
                continue

            high_level = prior_pipe.generate(
                c_embeds=neural_features[row].unsqueeze(0),
                num_inference_steps=prior_steps,
                guidance_scale=image_guidance_scale,
            )
            if generator is None:
                torch.save(high_level.detach().cpu(), path)
            else:
                image = generator.generate(high_level, generator=torch_generator)
                image.save(path)
            generated += 1
            if print_every > 0 and (count == 1 or count % print_every == 0):
                print(f"Wrote {path}", flush=True)

    return {
        "modality": modality,
        "subject": subject,
        "n_outputs": count,
        "n_generated": generated,
        "n_skipped": skipped,
        "output_type": "high_level_embedding" if generator is None else "image",
        "output_dir": str(subject_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BrainFLORA image reconstruction.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="+", choices=["eeg", "meg", "fmri"], default=["eeg", "meg", "fmri"])
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset, e.g. sub-01 02")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--prior-steps", type=int, default=10)
    parser.add_argument("--image-steps", type=int, default=4)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit per subject.")
    parser.add_argument("--sdxl-model", default="stabilityai/sdxl-turbo")
    parser.add_argument("--ip-adapter-model", default="h94/IP-Adapter")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Resume safely by skipping existing output files.")
    parser.add_argument("--print-every", type=int, default=25, help="Print every N generated outputs; use 1 for verbose.")
    parser.add_argument(
        "--skip-image-generator",
        action="store_true",
        help="Run encoder and reconstruction prior only, saving high-level embeddings instead of SDXL images.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reconstruction")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs/reconstruction/reconstruction_summary.json")
    parser.add_argument("--eeg-data-path", type=Path, default=DEFAULT_DATA_PATHS["eeg"])
    parser.add_argument("--meg-data-path", type=Path, default=DEFAULT_DATA_PATHS["meg"])
    parser.add_argument("--fmri-data-path", type=Path, default=DEFAULT_DATA_PATHS["fmri"])
    parser.add_argument("--eeg-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["eeg"])
    parser.add_argument("--meg-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["meg"])
    parser.add_argument("--fmri-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIRS["fmri"])
    parser.add_argument("--eeg-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["eeg"])
    parser.add_argument("--meg-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["meg"])
    parser.add_argument("--fmri-checkpoint", type=Path, default=DEFAULT_SINGLE_CHECKPOINTS["fmri"])
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_RECONSTRUCTION_CHECKPOINT)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
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
    model = build_unified_model(device, single_checkpoints, resolve_path(args.encoder_checkpoint), args.depth)
    prior_pipe = build_prior_pipe(device, resolve_path(args.prior_checkpoint))
    generator = (
        None
        if args.skip_image_generator
        else Generator4Embeds(
            num_inference_steps=args.image_steps,
            device=device,
            sdxl_model=args.sdxl_model,
            ip_adapter_model=args.ip_adapter_model,
            local_files_only=args.local_files_only,
        )
    )
    output_dir = resolve_path(args.output_dir)
    summary = []

    for modality in args.modalities:
        subjects = [normalize_subject(s) for s in args.subjects] if args.subjects else SUBJECTS[modality]
        subjects = [sub for sub in subjects if sub in SUBJECTS[modality]]
        for subject in subjects:
            print(f"Reconstructing modality={modality} subject={subject}")
            summary.append(
                reconstruct_subject(
                    model=model,
                    prior_pipe=prior_pipe,
                    generator=generator,
                    modality=modality,
                    subject=subject,
                    data_paths=data_paths,
                    device=device,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    prior_steps=args.prior_steps,
                    image_guidance_scale=args.image_guidance_scale,
                    seed=args.seed,
                    max_images=args.max_images,
                    print_every=args.print_every,
                    skip_existing=args.skip_existing,
                )
            )

    summary_path = resolve_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
