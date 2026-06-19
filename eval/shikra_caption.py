"""Generate BrainFLORA captions with Shikra/LLaMA.

This script formalizes the Shikra caption-generation path used by BrainFLORA:

    BrainFLORA token embeddings [N, 256, 1024]
        -> external ``mm_projector.bin`` [1024 -> LLaMA hidden size]
        -> replace Shikra ``<im_patch>`` token embeddings
        -> ``LlamaForCausalLM.generate``

Outputs follow the BrainFLORA caption layout:

    Caption/<MODALITY>_caption/Prior/shikra_<modality>_sub_<subject>_caption.{txt,json}
    Caption/<MODALITY>_caption/woPrior/shikra_<modality>_sub_<subject>_caption.{txt,json}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUBJECTS = {
    "eeg": ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10"),
    "meg": ("01", "02", "03", "04"),
    "fmri": ("01", "02", "03"),
}
DEFAULT_IMAGE_DIRS = {
    "eeg": Path("/vePFS-0x0d/visual/dataset/THINGS_EEG/images_set/test_images"),
    "meg": Path("/vePFS-0x0d/visual/dataset/THINGS_MEG/images_set_filter/test_images"),
    "fmri": Path("/vePFS-0x0d/visual/dataset/fmri_dataset/images/test_images"),
}
FEATURE_MODALITY_DIRS = {
    "eeg": "EEG",
    "meg": "MEG",
    "fmri": "fMRI",
}
OUTPUT_MODALITY_DIRS = {
    "eeg": "EEG_caption",
    "meg": "MEG_caption",
    "fmri": "fMRI_caption",
}
OUTPUT_MODALITY_NAMES = {
    "eeg": "eeg",
    "meg": "meg",
    "fmri": "fMRI",
}
VALID_MODALITIES = tuple(DEFAULT_SUBJECTS)
VALID_CONDITIONS = ("Prior", "woPrior")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@dataclass(frozen=True)
class CaptionTask:
    modality: str
    subject: str
    condition: str
    embedding_path: Path
    image_dir: Path
    output_dir: Path
    output_prefix: str
    output_suffix: str = ""

    @property
    def txt_path(self) -> Path:
        tag = OUTPUT_MODALITY_NAMES[self.modality]
        return self.output_dir / (
            f"{self.output_prefix}_{tag}_sub_{self.subject}_caption{self.output_suffix}.txt"
        )

    @property
    def json_path(self) -> Path:
        tag = OUTPUT_MODALITY_NAMES[self.modality]
        return self.output_dir / (
            f"{self.output_prefix}_{tag}_sub_{self.subject}_caption{self.output_suffix}.json"
        )


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_subject(subject: str) -> str:
    subject = subject.removeprefix("sub-").removeprefix("sub")
    if not subject.isdigit():
        raise ValueError(f"Invalid subject id: {subject!r}")
    return f"{int(subject):02d}"


def normalize_condition(condition: str) -> str:
    lowered = condition.lower()
    if lowered == "prior":
        return "Prior"
    if lowered in {"woprior", "wo_prior", "without_prior", "noprior", "no_prior"}:
        return "woPrior"
    raise ValueError(f"Invalid condition: {condition!r}. Use Prior or woPrior.")


def format_template(template: str, modality: str, subject: str, condition: str) -> Path:
    values = {
        "modality": modality,
        "MODALITY": modality.upper(),
        "Modality": modality.capitalize(),
        "subject": subject,
        "sub": subject,
        "condition": condition,
        "condition_lower": condition.lower(),
        "condition_dir": condition,
    }
    return Path(template.format(**values)).expanduser()


def default_embedding_candidates(
    embeddings_root: Path,
    modality: str,
    subject: str,
    condition: str,
) -> list[Path]:
    modality_dirs = [FEATURE_MODALITY_DIRS[modality]]
    if modality.upper() not in modality_dirs:
        modality_dirs.append(modality.upper())

    candidates: list[Path] = []
    for modality_dir in modality_dirs:
        candidates.extend(
            [
                embeddings_root
                / modality_dir
                / "256_1024"
                / f"FLORA_neural_features_sub-{subject}_{condition}_test.pt",
                embeddings_root
                / modality_dir
                / "256_1024"
                / f"FLORA_neural_features_sub-{subject}_{condition.lower()}_test.pt",
                embeddings_root
                / modality_dir
                / f"FLORA_neural_features_sub-{subject}_{condition}_test.pt",
                embeddings_root
                / modality_dir
                / f"FLORA_neural_features_sub-{subject}_{condition.lower()}_test.pt",
            ]
        )
    return candidates


def resolve_embedding_path(
    *,
    embeddings_root: Path,
    embedding_pattern: str | None,
    modality: str,
    subject: str,
    condition: str,
) -> Path:
    if embedding_pattern:
        path = format_template(embedding_pattern, modality, subject, condition)
        return path if path.is_absolute() else PROJECT_ROOT / path

    candidates = default_embedding_candidates(embeddings_root, modality, subject, condition)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    images.sort(key=lambda path: (path.parent.name, path.stem, path.name))
    if not images:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    return images


def align_images_to_embeddings(images: list[Path], n_embeddings: int) -> list[Path]:
    if len(images) == n_embeddings:
        return images
    if len(images) > n_embeddings and len(images) % n_embeddings == 0:
        stride = len(images) // n_embeddings
        return images[::stride]
    raise ValueError(
        f"Image count ({len(images)}) does not match embedding count ({n_embeddings}). "
        "Pass the correct --image-dir or adjust the saved embeddings."
    )


def torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_embeddings(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file does not exist: {path}")
    embeddings = torch_load(path, map_location="cpu")
    if isinstance(embeddings, dict):
        for key in ("features", "neural_features", "embeddings", "image_embedding"):
            if key in embeddings:
                embeddings = embeddings[key]
                break
        else:
            raise ValueError(
                f"Embedding dict in {path} does not contain one of: "
                "features, neural_features, embeddings, image_embedding"
            )
    if not torch.is_tensor(embeddings):
        raise TypeError(f"Expected tensor embeddings in {path}, got {type(embeddings)!r}")
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(1)
    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected embeddings shaped [N, tokens, dim] or [N, dim], "
            f"got {tuple(embeddings.shape)} from {path}"
        )
    return embeddings.float()


def normalize_projector_state(state: object) -> dict[str, torch.Tensor]:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected mm_projector state dict, got {type(state)!r}")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        short_key = key.split(".")[-1]
        if short_key in {"weight", "bias"}:
            normalized[short_key] = value
    if "weight" not in normalized:
        raise ValueError("mm_projector state does not contain a linear weight tensor.")
    return normalized


def load_mm_projector(path: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Linear:
    if not path.exists():
        raise FileNotFoundError(f"mm_projector checkpoint does not exist: {path}")
    state = normalize_projector_state(torch_load(path, map_location="cpu"))
    weight = state["weight"]
    if weight.ndim != 2:
        raise ValueError(f"Expected projector weight [out_dim, in_dim], got {tuple(weight.shape)}")
    out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])
    projector = torch.nn.Linear(in_dim, out_dim, bias="bias" in state)
    projector.load_state_dict(state, strict=True)
    projector.to(device=device, dtype=dtype)
    projector.eval()
    for param in projector.parameters():
        param.requires_grad_(False)
    return projector


def load_shikra_model(
    shikra_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
):
    if not shikra_path.exists() and local_files_only:
        raise FileNotFoundError(f"Shikra model path does not exist: {shikra_path}")
    try:
        from transformers import LlamaForCausalLM, LlamaTokenizer
    except ImportError as exc:
        raise ImportError(
            "Shikra captioning requires transformers with LLaMA support. "
            "Install transformers and sentencepiece in the active environment."
        ) from exc

    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            str(shikra_path),
            padding_side="left",
            local_files_only=local_files_only,
        )
    except ImportError as exc:
        raise ImportError(
            "LlamaTokenizer requires sentencepiece. Install it in the active environment, "
            "for example: pip install sentencepiece"
        ) from exc

    model = LlamaForCausalLM.from_pretrained(
        str(shikra_path),
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_prompt(prompt: str, num_patches: int) -> str:
    system = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    )
    user_image = " <im_start>" + "<im_patch>" * num_patches + "<im_end> "
    user_prompt = prompt.replace("<image>", user_image) if "<image>" in prompt else prompt + user_image
    return system + user_prompt + " ASSISTANT:"


def build_inputs_embeds(
    *,
    model,
    input_ids: torch.Tensor,
    projected_image_tokens: torch.Tensor,
    image_start_token_id: int,
    image_end_token_id: int,
) -> torch.Tensor:
    base_embeds = model.model.embed_tokens(input_ids)
    batch_embeds = base_embeds.repeat(projected_image_tokens.shape[0], 1, 1)
    outputs = []
    num_patches = projected_image_tokens.shape[1]
    for image_tokens, current_ids, current_embeds in zip(
        projected_image_tokens,
        input_ids.repeat(projected_image_tokens.shape[0], 1),
        batch_embeds,
        strict=True,
    ):
        image_start_positions = torch.where(current_ids == image_start_token_id)[0]
        if len(image_start_positions) != 1:
            raise ValueError(
                f"Expected exactly one <im_start> token id {image_start_token_id}, "
                f"found {len(image_start_positions)}."
            )
        image_start = int(image_start_positions[0].item())
        image_end = image_start + num_patches + 1
        if image_end >= current_ids.numel():
            raise ValueError("Input sequence is too short for the projected image features.")
        if int(current_ids[image_end].item()) != image_end_token_id:
            raise ValueError(
                f"The image end token should be at position {image_end} with id "
                f"{image_end_token_id}, got {int(current_ids[image_end].item())}."
            )
        outputs.append(
            torch.cat(
                (
                    current_embeds[: image_start + 1],
                    image_tokens,
                    current_embeds[image_end:],
                ),
                dim=0,
            )
        )
    return torch.stack(outputs, dim=0)


def clean_response(text: str, prompt_text: str) -> str:
    text = text.strip()
    for token in ("<s>", "</s>", "<unk>"):
        text = text.replace(token, "")
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[-1]
    elif prompt_text in text:
        text = text.split(prompt_text, 1)[-1]
    return " ".join(text.strip().split())


@torch.inference_mode()
def generate_captions(
    *,
    embeddings: torch.Tensor,
    projector: torch.nn.Linear,
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    prompt: str,
    generation_batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    image_start_token_id: int,
    image_end_token_id: int,
    pad_token_id: int | None,
    bos_token_id: int | None,
    eos_token_id: int | None,
) -> list[str]:
    num_patches = embeddings.shape[1]
    prompt_text = build_prompt(prompt, num_patches)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    captions: list[str] = []

    for start in range(0, embeddings.shape[0], generation_batch_size):
        batch = embeddings[start : start + generation_batch_size].to(device=device, dtype=dtype)
        projected = projector(batch)
        inputs_embeds = build_inputs_embeds(
            model=model,
            input_ids=input_ids,
            projected_image_tokens=projected,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
        )
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id if pad_token_id is not None else tokenizer.pad_token_id,
            bos_token_id=bos_token_id if bos_token_id is not None else tokenizer.bos_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        for offset, response in enumerate(decoded, start=1):
            idx = start + offset
            caption = clean_response(response, prompt_text)
            captions.append(caption)
            print(f"[{idx:04d}/{embeddings.shape[0]:04d}] {caption}", flush=True)
    return captions


def write_caption_outputs(task: CaptionTask, image_paths: list[Path], captions: list[str]) -> None:
    if len(image_paths) != len(captions):
        raise ValueError(
            f"Cannot write outputs: {len(image_paths)} images but {len(captions)} captions."
        )
    task.output_dir.mkdir(parents=True, exist_ok=True)
    task.txt_path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    task.json_path.write_text(
        json.dumps(
            {
                image_path.stem: caption
                for image_path, caption in zip(image_paths, captions, strict=True)
            },
            ensure_ascii=False,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {task.txt_path}")
    print(f"Wrote {task.json_path}")


def expand_tasks(args: argparse.Namespace) -> list[CaptionTask]:
    embeddings_root = resolve_path(args.embeddings_root)
    output_root = resolve_path(args.output_root)
    modalities = [modality.lower() for modality in args.modalities]
    conditions = [normalize_condition(condition) for condition in args.conditions]
    requested_subjects = [normalize_subject(subject) for subject in args.subjects] if args.subjects else None
    tasks: list[CaptionTask] = []
    for modality in modalities:
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality {modality!r}; choose from {VALID_MODALITIES}")
        subjects = requested_subjects or list(DEFAULT_SUBJECTS[modality])
        image_dir = resolve_path(args.image_dir) if args.image_dir else DEFAULT_IMAGE_DIRS[modality]
        for condition in conditions:
            condition_dir = "woPrior" if condition == "woPrior" else "Prior"
            for subject in subjects:
                embedding_path = resolve_embedding_path(
                    embeddings_root=embeddings_root,
                    embedding_pattern=args.embedding_pattern,
                    modality=modality,
                    subject=subject,
                    condition=condition,
                )
                tasks.append(
                    CaptionTask(
                        modality=modality,
                        subject=subject,
                        condition=condition,
                        embedding_path=embedding_path,
                        image_dir=image_dir,
                        output_dir=output_root / OUTPUT_MODALITY_DIRS[modality] / condition_dir,
                        output_prefix=args.output_prefix,
                        output_suffix=args.output_suffix,
                    )
                )
    return tasks


def print_dry_run(tasks: Iterable[CaptionTask], args: argparse.Namespace) -> None:
    shikra_path = resolve_path(args.shikra_path)
    projector_path = resolve_path(args.mm_projector_path)
    print(f"Shikra model: {'OK' if shikra_path.exists() else 'MISSING'} {shikra_path}")
    print(f"mm_projector: {'OK' if projector_path.exists() else 'MISSING'} {projector_path}")
    missing = 0
    for task in tasks:
        exists = "OK" if task.embedding_path.exists() else "MISSING"
        missing += int(not task.embedding_path.exists())
        print(
            f"{exists:7s} {task.modality.upper():4s} sub-{task.subject} "
            f"{task.condition:7s} -> {task.embedding_path}\n"
            f"        images -> {task.image_dir}\n"
            f"        output -> {task.json_path}"
        )
    if missing:
        raise SystemExit(f"Dry run found {missing} missing embedding file(s).")


def run_task(
    task: CaptionTask,
    *,
    args: argparse.Namespace,
    projector: torch.nn.Linear,
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if task.json_path.exists() and not args.overwrite:
        print(f"Skipping existing output: {task.json_path}")
        return
    print(
        f"\n=== {task.modality.upper()} sub-{task.subject} {task.condition} ===\n"
        f"embedding: {task.embedding_path}\n"
        f"images:    {task.image_dir}\n"
        f"output:    {task.output_dir}"
    )
    embeddings = load_embeddings(task.embedding_path)
    image_paths = align_images_to_embeddings(list_images(task.image_dir), len(embeddings))
    start = args.start_index
    stop = None if args.max_items is None else start + args.max_items
    embeddings = embeddings[start:stop]
    image_paths = image_paths[start:stop]
    if len(embeddings) == 0:
        raise ValueError(
            f"No embeddings selected for {task.embedding_path}; "
            f"start_index={args.start_index}, max_items={args.max_items}"
        )
    captions = generate_captions(
        embeddings=embeddings,
        projector=projector,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        prompt=args.prompt,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        image_start_token_id=args.image_start_token_id,
        image_end_token_id=args.image_end_token_id,
        pad_token_id=args.pad_token_id,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
    )
    write_caption_outputs(task, image_paths, captions)


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value.lower() == "none":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BrainFLORA captions with Shikra.")
    parser.add_argument("--modalities", nargs="+", default=list(VALID_MODALITIES), choices=VALID_MODALITIES)
    parser.add_argument("--conditions", nargs="+", default=list(VALID_CONDITIONS))
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional subject subset, e.g. 01 02 or sub-01.")
    parser.add_argument("--embeddings-root", default=str(PROJECT_ROOT / "features" / "FLORA"))
    parser.add_argument(
        "--embedding-pattern",
        default=None,
        help=(
            "Optional .pt path template. Available fields: {modality}, {MODALITY}, "
            "{subject}, {condition}, {condition_lower}, {condition_dir}."
        ),
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "Caption"))
    parser.add_argument("--output-prefix", default="shikra")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--image-dir", default=None, help="Override image directory for all selected tasks.")
    parser.add_argument("--shikra-path", default=str(PROJECT_ROOT / "external_models" / "shikra-7b"))
    parser.add_argument("--mm-projector-path", default=str(PROJECT_ROOT / "external_models" / "mm_projector.bin"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default="Describe this image <image> as simply as possible.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--image-start-token-id", type=int, default=32001)
    parser.add_argument("--image-end-token-id", type=int, default=32002)
    parser.add_argument("--pad-token-id", type=parse_optional_int, default="2")
    parser.add_argument("--bos-token-id", type=parse_optional_int, default="1")
    parser.add_argument("--eos-token-id", type=parse_optional_int, default="2")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned tasks without loading Shikra.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = expand_tasks(args)
    if args.dry_run:
        print_dry_run(tasks, args)
        return

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype, device)
    projector = load_mm_projector(resolve_path(args.mm_projector_path), device, dtype)
    model, tokenizer = load_shikra_model(
        resolve_path(args.shikra_path),
        device,
        dtype,
        local_files_only=args.local_files_only,
    )
    for task in tasks:
        run_task(
            task,
            args=args,
            projector=projector,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )


if __name__ == "__main__":
    main()
