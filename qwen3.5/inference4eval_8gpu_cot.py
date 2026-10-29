#!/usr/bin/env python
# coding=utf-8
"""Multi-GPU Qwen3.5 inference: caption -> complete annotation JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None


mp.set_start_method("spawn", force=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER = "{{此处替换为实际的Caption文本}}"
USER_PROMPT = "请根据上述要求完成标注，只输出一个符合指定结构的合法 JSON 对象，不要输出解释、注释或 Markdown 代码块。"
DEFAULT_MODEL_PATH = "/apdcephfs_gy2_302507476/share_302507476/xiaodayang/AudioSummary/CaptionSummary/ckpt/qwen3.5_caption"
DEFAULT_INPUT_CSV = "/apdcephfs_gy7/share_303660325/hunyuan/platform/hy_dp/user/jamesjzhou/20260818/delivery_57597d0ffc004ef78d3868375fecd4a4/gemini_file/00000.csv"
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "audio_summary", "qwen_prediction.csv")
DEFAULT_SYS_PROMPT = os.path.join(PROJECT_ROOT, "sys_prompt.md")
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


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


def load_system_prompt(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"system prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read()
    if PLACEHOLDER not in prompt:
        raise ValueError(f"system prompt must contain placeholder: {PLACEHOLDER}")
    return prompt


def build_messages(system_prompt: str, caption: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.replace(PLACEHOLDER, caption)},
        {"role": "user", "content": USER_PROMPT},
    ]


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Current tokenizer/transformers does not support enable_thinking=False. "
            "Please use the Qwen3.5-compatible transformers environment."
        ) from exc


def extract_json_text(text: str) -> str:
    value = (text or "").strip()
    match = JSON_FENCE.search(value)
    if match:
        value = match.group(1).strip()
    start = value.find("{")
    if start < 0:
        return value
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return value


def normalize_prediction(text: str) -> Tuple[str, bool, str]:
    cleaned = extract_json_text(text)
    try:
        obj = json.loads(cleaned)
    except Exception as exc:
        return cleaned, False, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return cleaned, False, "JSON top level is not an object"
    elements = obj.get("content_elements")
    content_types = obj.get("content_types")
    errors = []
    if not isinstance(elements, dict):
        errors.append("content_elements is not an object")
    else:
        missing_elements = [
            key for key in ("Environment", "Speaker", "Music", "SoundEffects", "Content")
            if key not in elements
        ]
        if missing_elements:
            errors.append(f"content_elements missing: {missing_elements}")
    if not isinstance(content_types, dict):
        errors.append("content_types is not an object")
    else:
        missing_types = [
            key for key in ("category_1", "category_2", "category_3", "type_summary")
            if key not in content_types
        ]
        if missing_types:
            errors.append(f"content_types missing: {missing_types}")
    normalized = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if errors:
        return normalized, False, "; ".join(errors)
    return normalized, True, ""


def read_csv_rows(path: str, caption_field: str, max_samples: int) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if caption_field not in fields:
            raise ValueError(f"caption field {caption_field!r} not found; available={fields}")
        for row in reader:
            rows.append(dict(row))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"no samples in {path}")
    return rows, fields


def output_fields(input_fields: List[str]) -> List[str]:
    result = list(input_fields)
    for name in ("qwen_result", "qwen_json_valid", "qwen_error", "pred_time_sec", "qwen_gpu"):
        if name not in result:
            result.append(name)
    return result


def atomic_write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".qwen_prediction.", suffix=".csv", dir=output_dir)
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_completed_indices(path: str, total: int) -> Dict[int, Dict[str, str]]:
    completed: Dict[int, Dict[str, str]] = {}
    if not os.path.isfile(path):
        return completed
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for index, row in enumerate(reader):
                if index >= total:
                    break
                if str(row.get("qwen_result", "") or "").strip() or str(row.get("qwen_error", "") or "").strip():
                    completed[index] = dict(row)
    except Exception as exc:
        print(f"[WARN] cannot load resume output {path}: {exc}", flush=True)
    return completed


def load_model_on_gpu(gpu_id: int, model_path: str, trust_remote_code: bool):
    if Qwen3_5ForConditionalGeneration is None:
        raise RuntimeError("Qwen3_5ForConditionalGeneration is unavailable in this transformers installation")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def infer_one(tokenizer, model, device, caption: str, system_prompt: str, max_new_tokens: int) -> str:
    prompt = apply_chat_template(tokenizer, build_messages(system_prompt, caption))
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()


def worker(gpu_id: int, task_q: mp.Queue, result_q: mp.Queue, config: Dict[str, Any]) -> None:
    try:
        tokenizer, model, device = load_model_on_gpu(
            gpu_id, config["model_path"], config["trust_remote_code"]
        )
        result_q.put(("ready", gpu_id))
    except Exception:
        result_q.put(("fatal", gpu_id, traceback.format_exc()))
        return

    while True:
        task = task_q.get()
        if task is None:
            break
        index, caption = task
        started = time.time()
        try:
            raw = infer_one(
                tokenizer,
                model,
                device,
                caption,
                config["system_prompt"],
                config["max_new_tokens"],
            )
            prediction, valid, error = normalize_prediction(raw)
        except Exception as exc:
            prediction, valid, error = "", False, f"{type(exc).__name__}: {exc}"
        result_q.put(("result", index, prediction, valid, error, time.time() - started, gpu_id))


def parse_gpu_ids(value: str) -> List[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus is empty")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"duplicate GPU IDs: {gpu_ids}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    count = torch.cuda.device_count()
    invalid = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= count]
    if invalid:
        raise ValueError(f"invalid GPU IDs {invalid}; visible device count={count}")
    return gpu_ids


def stop_workers(processes: List[mp.Process], terminate: bool) -> None:
    for process in processes:
        if terminate and process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.is_alive():
            process.join(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8-GPU Qwen3.5 caption annotation inference")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--caption_field", default="caption")
    parser.add_argument("--sys_prompt", default=DEFAULT_SYS_PROMPT)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--worker_start_timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Qwen3_5ForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen3_5ForConditionalGeneration is unavailable. Activate the Qwen3.5 environment "
            "or install a compatible transformers version."
        )
    gpu_ids = parse_gpu_ids(args.gpus)
    model_path = resolve_model_path(args.model_path)
    system_prompt = load_system_prompt(args.sys_prompt)
    input_path = os.path.abspath(os.path.expanduser(args.input_csv))
    output_path = os.path.abspath(os.path.expanduser(args.output_csv))
    rows, input_fields = read_csv_rows(input_path, args.caption_field, args.max_samples)
    fields = output_fields(input_fields)

    completed = load_completed_indices(output_path, len(rows)) if args.resume else {}
    for index, saved in completed.items():
        for field in fields:
            if field in saved:
                rows[index][field] = saved[field]
    pending = [index for index in range(len(rows)) if index not in completed]
    print(
        f"Input={len(rows)} completed={len(completed)} pending={len(pending)} GPUs={gpu_ids}",
        flush=True,
    )
    if not pending:
        atomic_write_csv(output_path, rows, fields)
        print(f"All samples already completed: {output_path}", flush=True)
        return

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    config = {
        "model_path": model_path,
        "system_prompt": system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "trust_remote_code": args.trust_remote_code,
    }
    processes = [mp.Process(target=worker, args=(gpu, task_q, result_q, config)) for gpu in gpu_ids]
    for process in processes:
        process.start()

    ready: Set[int] = set()
    deadline = time.time() + args.worker_start_timeout
    try:
        while len(ready) < len(processes):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"workers did not become ready: ready={sorted(ready)}")
            try:
                message = result_q.get(timeout=min(10, remaining))
            except queue.Empty:
                dead = [(gpu, process.exitcode) for gpu, process in zip(gpu_ids, processes) if not process.is_alive()]
                if dead:
                    raise RuntimeError(f"workers exited during model loading: {dead}")
                continue
            if message[0] == "fatal":
                raise RuntimeError(f"worker GPU {message[1]} failed to load:\n{message[2]}")
            if message[0] == "ready":
                ready.add(int(message[1]))
        print(f"All {len(ready)} workers ready", flush=True)

        def produce() -> None:
            for index in pending:
                caption = str(rows[index].get(args.caption_field, "") or "")
                task_q.put((index, caption))
            for _ in processes:
                task_q.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        done = 0
        with tqdm(total=len(pending), desc="infer", unit="sample") as progress:
            while done < len(pending):
                try:
                    message = result_q.get(timeout=30)
                except queue.Empty:
                    dead = [(gpu, process.exitcode) for gpu, process in zip(gpu_ids, processes) if not process.is_alive()]
                    if dead and done < len(pending):
                        raise RuntimeError(f"workers exited before all results arrived: {dead}, done={done}")
                    continue
                if message[0] == "fatal":
                    raise RuntimeError(f"worker GPU {message[1]} failed:\n{message[2]}")
                if message[0] != "result":
                    continue
                _, index, prediction, valid, error, elapsed, gpu_id = message
                rows[index]["qwen_result"] = prediction
                rows[index]["qwen_json_valid"] = str(bool(valid)).lower()
                rows[index]["qwen_error"] = error
                rows[index]["pred_time_sec"] = f"{elapsed:.6f}"
                rows[index]["qwen_gpu"] = str(gpu_id)
                done += 1
                progress.update(1)
                if args.checkpoint_interval > 0 and done % args.checkpoint_interval == 0:
                    atomic_write_csv(output_path, rows, fields)
        producer.join(timeout=10)
        stop_workers(processes, terminate=False)
        atomic_write_csv(output_path, rows, fields)
        print(f"Wrote {len(rows)} rows to {output_path}", flush=True)
    except Exception:
        atomic_write_csv(output_path, rows, fields)
        stop_workers(processes, terminate=True)
        raise


if __name__ == "__main__":
    main()
