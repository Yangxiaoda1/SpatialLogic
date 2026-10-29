#!/usr/bin/env python
# coding=utf-8
"""Qwen3.5 multimodal SFT for spatial task-progress reasoning and planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoProcessor, Trainer, TrainingArguments

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

try:
    from accelerate.utils import set_seed
except Exception:
    def set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/mkxlsx_train/ckpt/qwen3.5_9B"
DEFAULT_DATA_ROOT = "/apdcephfs_gy5/share_302507476/xiaodayang/SpatialLogic_Data"
DEFAULT_OUTPUT_DIR = "/apdcephfs_gy2_302507476/share_302507476/xiaodayang/AudioSummary/SpatialLogic/ckpt/qwen3.5_spatiallogic"
DEFAULT_MISSING_DIR = str(PROJECT_ROOT / "missing")
DEFAULT_BROKEN_DIR = str(PROJECT_ROOT / "missing" / "broken")
DEFAULT_TRAIN_FILES = [
    str(PROJECT_ROOT / "training_data/equal/equal_ego_ego4d.json"),
    str(PROJECT_ROOT / "training_data/equal/equal_ego_epic.json"),
    str(PROJECT_ROOT / "training_data/equal/equal_short_trainset.json"),
    str(PROJECT_ROOT / "training_data/leap/all_168042.jsonl"),
    str(PROJECT_ROOT / "training_data/short/tag50_finished.json"),
    str(PROJECT_ROOT / "training_data/short/cot50fin_cleaned_filtered_translated.json"),
    str(PROJECT_ROOT / "training_data/tag_long_ego/all_train.jsonl"),
]
SYSTEM_PROMPT = """You are an expert in visual spatial perception, task-progress reasoning, and task planning.

For two-image task-progress samples, compare img1 and img2 from the same task using visible evidence and the task description. Choose img1 if img1 is closer to completion, img2 if img2 is closer, or equal if both images show the same or indistinguishably close task-completion state.

For single-image task-planning samples, infer the ordered steps needed to complete the task from the visible state and the task description.

Rules:
1. Treat image identifiers exactly as given in the user message; never swap img1 and img2.
2. Base judgments only on visible task-relevant evidence. Do not assume that a later frame is necessarily closer to completion.
3. If spatial details are requested, describe each image separately and make the final judgment consistent with those details.
4. Follow the output format requested by the user exactly.
5. For closer to completion, use only img1, img2, or equal.
6. Do not add Markdown fences, headings, disclaimers, or commentary outside the requested output fields."""


def load_system_prompt(path: Optional[str]) -> str:
    if path:
        resolved_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"system prompt not found: {resolved_path}")
        with open(resolved_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        source = resolved_path
    else:
        prompt = SYSTEM_PROMPT
        source = "built-in spatial system prompt"
    if not prompt:
        raise ValueError("system prompt must not be empty")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    print(f"System prompt: {source} sha256={digest}", flush=True)
    return prompt


def apply_chat_template(processor, messages: List[Dict[str, Any]]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def resolve_model_path(model_path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(model_path)) if os.path.isabs(model_path) else model_path
    if os.path.isfile(os.path.join(expanded, "config.json")):
        return expanded
    snapshots = os.path.join(expanded, "snapshots")
    if os.path.isdir(snapshots):
        candidates = [os.path.join(snapshots, name) for name in os.listdir(snapshots)]
        candidates = [path for path in candidates if os.path.isfile(os.path.join(path, "config.json"))]
        if candidates:
            return max(candidates, key=os.path.getmtime)
    if os.path.isabs(model_path) and not os.path.exists(expanded):
        raise FileNotFoundError(f"model path not found: {expanded}")
    return model_path


def iter_json_records(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    array_mode = False
    eof = False
    with path.open("r", encoding="utf-8") as f:
        while True:
            if not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if not started:
                    started = True
                    if buffer.startswith("["):
                        array_mode = True
                        buffer = buffer[1:]
                        continue
                if array_mode:
                    buffer = buffer.lstrip()
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        continue
                    if buffer.startswith("]"):
                        return
                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if isinstance(obj, dict):
                    yield obj
                buffer = buffer[end:]
            if eof:
                if buffer.strip() in ("", "]"):
                    return
                raise json.JSONDecodeError("trailing JSON data", buffer, 0)


def normalize_target_key(key: str) -> str:
    return "closer to completion" if key == "closer_to_completion" else key


def stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def serialize_target(target: Any) -> str:
    if isinstance(target, str):
        stripped = target.strip()
        try:
            target = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    if not isinstance(target, dict):
        return stringify_value(target)

    normalized = {normalize_target_key(str(k)): v for k, v in target.items()}
    preferred_order = ("spatial details", "closer to completion", "steps")
    keys = [key for key in preferred_order if key in normalized]
    keys.extend(key for key in normalized if key not in keys)
    return "\n".join(f"{key}: {stringify_value(normalized[key])}" for key in keys)


def fix_short_task_dir(path: Path) -> Optional[Path]:
    parent = path.parent
    if parent.name.endswith("_"):
        fixed = parent.with_name(parent.name[:-1] + ".") / path.name
        if fixed.exists():
            return fixed
    return None


def resolve_image_path(image: str, data_root: Path) -> Path:
    raw = str(image).strip()
    if not raw:
        raise FileNotFoundError("empty image path")
    path = Path(raw)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(raw)

    candidates: List[Path] = []
    parts = path.parts
    if raw.startswith("image/"):
        candidates.append(data_root / "images" / Path(*parts[1:]))
    elif raw.startswith("data/images/"):
        name = Path(*parts[2:]).name
        stem = Path(name).stem
        match = re.fullmatch(r"(\d+)-(\d+)", stem)
        if match:
            candidates.append(data_root / "images" / match.group(1) / match.group(2) / "00000.png")
        candidates.append(data_root / "images" / Path(*parts[2:]))
        candidates.append(data_root / "leap-agibot-processed" / Path(*parts[1:]))
    elif raw.startswith("trainset/"):
        candidates.append(data_root / "short" / path)
    elif raw.startswith("trainset_tasks/"):
        candidates.append(data_root / "short" / "trainset" / Path(*parts[1:]))
    elif raw.startswith("ego4d/") or raw.startswith("epic-100/"):
        candidates.append(data_root / "ego" / path)
    elif raw.startswith("images/"):
        rel = Path(*parts[1:])
        first = parts[1] if len(parts) > 1 else ""
        filename = parts[-1] if len(parts) > 2 else ""
        if "_part" in first:
            video_id = first.split("_part", 1)[0]
            candidates.extend([
                data_root / "ego" / "ego4d" / video_id / first / filename,
                data_root / "ego" / "ego4d" / first / filename,
            ])
        epic_match = re.fullmatch(r"(P\d{2})_.*", first)
        if epic_match:
            participant = epic_match.group(1)
            candidates.extend([
                data_root / "ego" / "epic-100" / participant / first / filename,
                data_root / "ego" / "epic-100" / first / filename,
            ])
        candidates.extend([
            data_root / "equal" / path,
            data_root / path,
            data_root / "ego" / rel,
            data_root / "ego" / "ego4d" / rel,
            data_root / "ego" / "epic-100" / rel,
        ])
    candidates.append(data_root / path)

    checked: List[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
        fixed = fix_short_task_dir(candidate)
        if fixed is not None:
            return fixed
    raise FileNotFoundError(f"image not found: {raw}; checked={checked[:6]}")


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class SampleTooLongError(ValueError):
    pass


def get_shard() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    worker = get_worker_info()
    if worker is None:
        return rank, world_size
    return rank * worker.num_workers + worker.id, world_size * worker.num_workers


class SpatialLogicIterableDataset(IterableDataset):
    def __init__(
        self,
        processor,
        train_files: List[str],
        data_root: str,
        system_prompt: str,
        max_samples: int,
        max_length: int,
        skip_missing_images: bool,
        missing_dir: str,
        broken_dir: str,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.train_files = [Path(path).expanduser().resolve() for path in train_files]
        self.data_root = Path(data_root).expanduser().resolve()
        self.system_prompt = system_prompt
        self.max_samples = int(max_samples)
        self.max_length = int(max_length)
        self.skip_missing_images = bool(skip_missing_images)
        self.missing_dir = Path(missing_dir).expanduser().resolve()
        self.broken_dir = Path(broken_dir).expanduser().resolve()
        for path in self.train_files:
            if not path.is_file():
                raise FileNotFoundError(f"training file not found: {path}")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data root not found: {self.data_root}")
        print("Training files:", flush=True)
        for path in self.train_files:
            print(f"  - {path}", flush=True)
        print(f"Data root: {self.data_root}", flush=True)
        print(f"Missing-image log dir: {self.missing_dir}", flush=True)
        print(f"Broken-sample log dir: {self.broken_dir}", flush=True)

    def _missing_log_path(self, shard_id: int) -> Path:
        return self.missing_dir / f"missing_rank{shard_id:05d}.jsonl"

    def _broken_log_path(self, shard_id: int) -> Path:
        return self.broken_dir / f"broken_rank{shard_id:05d}.jsonl"

    def _collect_image_refs(self, record: Dict[str, Any]) -> List[str]:
        image_refs: List[str] = []
        for message in record.get("messages", []):
            content = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and (item.get("type") == "image" or "image" in item):
                    image_refs.append(str(item.get("image", "")))
        return image_refs

    def _write_issue_log(
        self,
        log_path: Path,
        source_file: Path,
        sample_index: int,
        record: Dict[str, Any],
        error: BaseException,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_file": str(source_file),
            "sample_index": sample_index,
            "images": self._collect_image_refs(record),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _messages_and_images(self, record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        record_messages = record.get("messages")
        if not isinstance(record_messages, list):
            raise ValueError("record.messages must be a list")
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        images: List[Image.Image] = []
        for message in record_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                new_content: List[Dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image" or "image" in item:
                        image_path = resolve_image_path(str(item.get("image", "")), self.data_root)
                        images.append(load_image(image_path))
                        new_item = dict(item)
                        new_item["image"] = str(image_path)
                        new_content.append(new_item)
                    elif item.get("type") == "text" or "text" in item:
                        new_content.append({"type": "text", "text": str(item.get("text", ""))})
                messages.append({"role": role, "content": new_content})
            else:
                messages.append({"role": role, "content": str(content)})
        return messages, images

    def _encode(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        target = serialize_target(record.get("target", ""))
        if not target:
            raise ValueError("empty target")
        messages, images = self._messages_and_images(record)
        prompt = apply_chat_template(self.processor, messages)
        eos = self.tokenizer.eos_token or ""
        full_text = prompt + target + eos
        image_arg = images if images else None
        prompt_inputs = self.processor(text=[prompt], images=image_arg, return_tensors="pt")
        full_inputs = self.processor(text=[full_text], images=image_arg, return_tensors="pt")
        input_ids = full_inputs["input_ids"][0]
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        if self.max_length > 0 and input_ids.numel() > self.max_length:
            raise SampleTooLongError(f"sample length {input_ids.numel()} exceeds max_length={self.max_length}")
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        feature: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"][0],
            "labels": labels,
        }
        for key, value in full_inputs.items():
            if key in feature or not isinstance(value, torch.Tensor):
                continue
            if key == "mm_token_type_ids" and value.ndim == 2 and value.shape[0] == 1:
                feature[key] = value[0]
            else:
                feature[key] = value
        return feature

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        shard_id, num_shards = get_shard()
        missing_log_path = self._missing_log_path(shard_id)
        broken_log_path = self._broken_log_path(shard_id)
        seen = 0
        yielded = 0
        skipped_missing = 0
        skipped_broken = 0
        for path in self.train_files:
            for record in iter_json_records(path):
                if self.max_samples > 0 and seen >= self.max_samples:
                    return
                sample_index = seen
                seen += 1
                if sample_index % num_shards != shard_id:
                    continue
                try:
                    feature = self._encode(record)
                except FileNotFoundError as error:
                    skipped_missing += 1
                    self._write_issue_log(missing_log_path, path, sample_index, record, error)
                    if self.skip_missing_images:
                        if skipped_missing % 100 == 0:
                            print(
                                f"rank_shard={shard_id}/{num_shards} yielded={yielded} "
                                f"skipped_missing={skipped_missing} skipped_broken={skipped_broken} seen={seen} "
                                f"missing_log={missing_log_path}",
                                flush=True,
                            )
                        continue
                    raise
                except SampleTooLongError as error:
                    skipped_broken += 1
                    self._write_issue_log(broken_log_path, path, sample_index, record, error)
                    if skipped_broken % 100 == 0:
                        print(
                            f"rank_shard={shard_id}/{num_shards} yielded={yielded} "
                            f"skipped_missing={skipped_missing} skipped_broken={skipped_broken} seen={seen} "
                            f"broken_log={broken_log_path}",
                            flush=True,
                        )
                    continue
                except Exception as error:
                    skipped_broken += 1
                    self._write_issue_log(broken_log_path, path, sample_index, record, error)
                    print(
                        f"rank_shard={shard_id}/{num_shards} broken_sample={skipped_broken} "
                        f"seen={seen} broken_log={broken_log_path} error={type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                yielded += 1
                if yielded % 100 == 0:
                    print(
                        f"rank_shard={shard_id}/{num_shards} yielded={yielded} "
                        f"skipped_missing={skipped_missing} skipped_broken={skipped_broken} seen={seen}",
                        flush=True,
                    )
                yield feature


@dataclass
class SFTCollator:
    tokenizer: Any

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].numel() for item in batch)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        def pad_1d(tensor: torch.Tensor, value: int) -> torch.Tensor:
            if tensor.numel() == max_len:
                return tensor
            suffix = torch.full((max_len - tensor.numel(),), value, dtype=tensor.dtype)
            return torch.cat((tensor, suffix), dim=0)

        output: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack([pad_1d(item["input_ids"], pad_id) for item in batch]),
            "attention_mask": torch.stack([pad_1d(item["attention_mask"], 0) for item in batch]),
            "labels": torch.stack([pad_1d(item["labels"], -100) for item in batch]),
        }
        if "mm_token_type_ids" in batch[0]:
            output["mm_token_type_ids"] = torch.stack(
                [pad_1d(item.get("mm_token_type_ids", torch.zeros_like(item["input_ids"])), 0) for item in batch]
            )
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            values = [item[key] for item in batch if key in item]
            if values:
                output[key] = torch.cat(values, dim=0)
        return output


def count_or_freeze_parameters(model, freeze_visual: bool) -> Tuple[int, int, int]:
    total = trainable = frozen_visual = 0
    visual_markers = ("visual", "vision", "image")
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if freeze_visual and any(marker in name.lower() for marker in visual_markers):
            parameter.requires_grad = False
            frozen_visual += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable, frozen_visual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 multimodal SFT: SpatialLogic JSON/JSONL")
    parser.add_argument("--train_files", nargs="+", default=DEFAULT_TRAIN_FILES, help="Training JSON/JSONL file(s)")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Root directory containing images, short, ego, and equal data")
    parser.add_argument("--sys_prompt", default=None, help="Optional system-prompt text file")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "tb"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--freeze_visual", action="store_true", help="Freeze visual/vision/image parameters. Default: train them.")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--missing_dir", default=DEFAULT_MISSING_DIR, help="Directory for JSONL logs of missing-image samples")
    parser.add_argument("--broken_dir", default=DEFAULT_BROKEN_DIR, help="Directory for JSONL logs of broken samples")
    parser.add_argument(
        "--skip_missing_images",
        dest="skip_missing_images",
        action="store_true",
        default=True,
        help="Skip samples whose images are not available and log them to --missing_dir. This is the default.",
    )
    parser.add_argument(
        "--no_skip_missing_images",
        dest="skip_missing_images",
        action="store_false",
        help="Stop training when an image is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Qwen3_5ForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen3_5ForConditionalGeneration is unavailable. Activate the Qwen3.5 environment "
            "or install a compatible transformers version before training."
        )
    if args.max_steps <= 0:
        raise ValueError("streaming JSON/JSONL training requires --max_steps > 0")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.skip_missing_images:
        os.makedirs(args.missing_dir, exist_ok=True)
    os.makedirs(args.broken_dir, exist_ok=True)
    set_seed(args.seed)

    system_prompt = load_system_prompt(args.sys_prompt)
    model_path = resolve_model_path(args.model_path)
    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=False, trust_remote_code=args.trust_remote_code
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SpatialLogicIterableDataset(
        processor=processor,
        train_files=args.train_files,
        data_root=args.data_root,
        system_prompt=system_prompt,
        max_samples=args.max_samples,
        max_length=args.max_length,
        skip_missing_images=args.skip_missing_images,
        missing_dir=args.missing_dir,
        broken_dir=args.broken_dir,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    total, trainable, frozen_visual = count_or_freeze_parameters(model, args.freeze_visual)
    print(
        f"Parameters: total={total:,} trainable={trainable:,} "
        f"frozen_visual={frozen_visual:,} freeze_visual={args.freeze_visual}",
        flush=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        report_to="tensorboard",
        run_name=os.path.basename(os.path.abspath(args.output_dir)),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        remove_unused_columns=False,
        warmup_steps=args.warmup_steps if args.warmup_steps > 0 else args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_steps=args.max_steps,
        deepspeed=args.deepspeed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if trainer.is_world_process_zero():
        print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
