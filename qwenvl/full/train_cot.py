#!/usr/bin/env python
# coding=utf-8
"""
Train Qwen2.5-VL on CoT-style JSON data (images + prompt + analysis + closer_to_completion).

使用说明：
1. 修改文件顶部的路径配置区域，设置你的环境路径
2. 根据需要调整训练参数配置区域
3. 运行命令：python train_cot.py [可选参数]

可选参数：
  --cot_files: COT数据文件列表
  --base_dir: 项目根目录（包含testset文件夹）
  --model_path: 模型路径
  --output_dir: 输出目录
  --epochs: 训练轮数
  --batch_size: 批次大小
  --learning_rate: 学习率
  --max_samples: 最大样本数

JSON格式示例：
[
  {
    "messages": [{"role": "user", "content": [
        {"type": "image", "image": "testset\\...\\00101.jpg", "image_id": "img1"},
        {"type": "image", "image": "testset\\...\\00096.jpg", "image_id": "img2"},
        {"type": "text",   "text":   "...prompt..."}
    ]}],
    "target": {"analysis": "...", "closer_to_completion": "img1"}
  },
  ...
]
"""

# ==================== 路径配置区域 ====================
# 请根据你的环境修改以下路径
BASE_DIR = "/home/tione/notebook/Spaciallogic"  # 项目根目录
MODEL_PATH = "/home/tione/notebook/Spaciallogic/Qwen2.5-VL-7B-Instruct"  # 模型路径
OUTPUT_DIR = "/home/tione/notebook/Spaciallogic/qwenvl/full/cot_checkpoint"  # 输出目录
COT_FILES = ["new_forward.json", "new_reverse.json"]  # COT数据文件列表

# ==================== 训练参数配置区域 ====================
# 训练超参数，可根据需要调整
TRAINING_CONFIG = {
    "epochs": 1,                    # 训练轮数
    "batch_size": 1,                # 批次大小
    "learning_rate": 5e-6,          # 学习率
    "max_samples": 2000,            # 最大样本数（设为0表示使用全部数据）
    "max_steps": 2000,              # 最大训练步数
    "gradient_accumulation_steps": 8,  # 梯度累积步数
    "warmup_steps": 100,            # 预热步数
    "save_steps": 100,              # 保存检查点步数
    "logging_steps": 1,             # 日志记录步数
    "max_grad_norm": 0.01,          # 梯度裁剪阈值
    "weight_decay": 0.0001,         # 权重衰减
}

# LoRA配置参数
LORA_CONFIG = {
    "r": 16,                        # LoRA秩
    "lora_alpha": 32,               # LoRA缩放因子
    "lora_dropout": 0.05,           # LoRA dropout
    "target_modules": [              # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
}
# ===================================================

import os
# Reduce thread contention / avoid OpenBLAS OpenMP loop warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import argparse
from typing import List, Dict, Any

import torch
try:
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def resolve_path(p: str, base_dir: str) -> str:
    """Normalize a possibly Windows-style relative path, and try to fix trailing '_' dirs.
    base_dir: Directory from which to resolve relative paths (e.g., repo root containing 'testset').
    """
    if not p:
        return p
    # Normalize slashes
    p_norm = p.replace("\\", os.sep)
    if os.path.isabs(p_norm):
        cand = p_norm
    else:
        cand = os.path.join(base_dir, p_norm)
    if os.path.exists(cand):
        return cand
    # attempt replacing any component ending with '_' to '.'
    parts = cand.split(os.sep)
    for i in range(len(parts)):
        if parts[i].endswith('_'):
            alt_parts = parts.copy()
            alt_parts[i] = alt_parts[i][:-1] + '.'
            alt = os.sep.join(alt_parts)
            if os.path.exists(alt):
                return alt
    return cand  # return best-effort even if not exists; downstream will error if missing


class CoTDataset(Dataset):
    def __init__(self, processor, cot_files: List[str], base_dir: str, max_samples: int = None):
        self.processor = processor
        # Treat <=0 as unlimited
        if max_samples is not None and max_samples <= 0:
            max_samples = None
        self.samples: List[Dict[str, Any]] = []
        loaded = 0
        for jf in cot_files:
            jf_abs = jf if os.path.isabs(jf) else os.path.abspath(jf)
            if not os.path.exists(jf_abs):
                print(f"[WARN] CoT file not found: {jf_abs}")
                continue
            with open(jf_abs, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # data may be a single object or a list
            items = data if isinstance(data, list) else [data]
            for item in items:
                messages = item.get('messages', [])
                # Normalize image paths in-place
                for m in messages:
                    if 'content' in m and isinstance(m['content'], list):
                        for c in m['content']:
                            if isinstance(c, dict) and c.get('type') == 'image' and 'image' in c:
                                c['image'] = resolve_path(c['image'], base_dir)
                tgt = item.get('target', {})
                analysis = tgt.get('analysis', '')
                closer = tgt.get('closer_to_completion', tgt.get('closer to completion', ''))
                
                # 保留原始prompt，不要硬编码格式
                if analysis and closer:
                    # 直接使用JSON中的原始内容，不做格式转换
                    target_text = analysis + "\n" + closer
                else:
                    print(f"Warning: missing analysis or closer_to_completion in sample")
                    target_text = "No analysis available"
                self.samples.append({
                    'messages': messages,
                    'target': target_text
                })
                loaded += 1
                if max_samples is not None and loaded >= max_samples:
                    break
            if max_samples is not None and loaded >= max_samples:
                break
        if len(self.samples) == 0:
            print("[WARN] No samples loaded from CoT files.")
        else:
            print(f"Loaded {len(self.samples)} samples from CoT files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        target = sample['target']
        
        # 验证目标文本长度
        if len(target) > 1000:  # 限制目标文本长度
            target = target[:1000]
        
        # open images in order
        image_inputs = []
        for m in messages:
            for c in m.get('content', []):
                if isinstance(c, dict) and c.get('type') == 'image':
                    try:
                        img = Image.open(c['image']).convert('RGB')
                        image_inputs.append(img)
                    except Exception as e:
                        print(f"Error loading image {c['image']}: {e}")
                        # 如果图片加载失败，使用一个默认的1x1像素图片
                        img = Image.new('RGB', (1, 1), color='black')
                        image_inputs.append(img)
        
        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = chat_prompt + "\nassistant\n" + target
        
        # 重要：禁用truncation以保持多模态token对齐
        inputs = self.processor(
            text=full_text, 
            images=image_inputs, 
            return_tensors='pt', 
            padding=True, 
            truncation=False  # 禁用truncation避免图片token不匹配
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # mask out prompt tokens in labels
        prefix_len = self.processor.tokenizer(chat_prompt + "\nassistant\n", return_tensors='pt').input_ids.shape[-1] - 1
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        
        # 确保有足够的有效标签
        valid_label_count = (labels != -100).sum().item()
        if valid_label_count == 0:
            print(f"Warning: sample {idx} has no valid labels, setting last few tokens as valid")
            # 设置最后几个token为有效标签
            labels[-3:] = input_ids[-3:]  # 保留最后3个token作为标签
        
        # 处理pad token
        if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        # 最终验证
        final_valid_count = (labels != -100).sum().item()
        if final_valid_count == 0:
            print(f"Error: sample {idx} still has no valid labels after fixing")
            # 强制设置一些标签
            labels[-1] = input_ids[-1]
        
        # 添加样本信息调试
        if idx < 3:
            print(f"Sample {idx}: input_len={len(input_ids)}, prefix_len={prefix_len}, valid_labels={final_valid_count}")
            print(f"  Target length: {len(target)}")
            print(f"  Target preview: {target[:100]}...")
            print(f"  Labels: {labels.tolist()[-10:]}")  # 显示最后10个标签
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class GradientNaNCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
            
        # 检查NaN和Inf梯度
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad = p.grad.data
                if torch.isnan(grad).any():
                    print(f"NaN grad at {name}, replacing with zeros")
                    p.grad.data = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
                elif torch.isinf(grad).any():
                    print(f"Inf grad at {name}, replacing with zeros")
                    p.grad.data = torch.where(torch.isinf(grad), torch.zeros_like(grad), grad)
                
                # 使用args.max_grad_norm进行梯度裁剪
                grad_norm = grad.norm()
                if grad_norm > args.max_grad_norm:
                    print(f"Large grad at {name}: {grad_norm.item()}, clipping to {args.max_grad_norm}")
                    p.grad.data = torch.clamp(grad, -args.max_grad_norm, args.max_grad_norm)
                elif torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Invalid grad norm at {name}: {grad_norm.item()}, zeroing")
                    p.grad.data = torch.zeros_like(grad)


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen2.5-VL on CoT JSON")
    parser.add_argument("--cot_files", nargs="+", default=COT_FILES, help="One or more CoT JSON files (e.g., new_forward.json new_reverse.json)")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Base dir to resolve relative image paths (should contain testset)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Local model dir")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output dir")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG["learning_rate"])
    parser.add_argument("--max_samples", type=int, default=TRAINING_CONFIG["max_samples"], help="Limit samples to avoid OOM on CPU")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    dtype = torch.float32  # 强制使用float32提高数值稳定性
    print(f"Using CUDA: {use_cuda}, dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=dtype,
        use_cache=False,  # 禁用 KV 缓存节省显存
        low_cpu_mem_usage=True,  # 减少 CPU 内存使用
    )

    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    # 确保use_cache为False
    model.config.use_cache = False
    
    # 设置梯度缩放以提高数值稳定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 清理显存
    if use_cuda:
        torch.cuda.empty_cache()

    # 配置LoRA
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],  # 使用配置中的秩
        lora_alpha=LORA_CONFIG["lora_alpha"],  # 使用配置中的缩放因子
        target_modules=LORA_CONFIG["target_modules"],  # 使用配置中的目标模块
        lora_dropout=LORA_CONFIG["lora_dropout"],  # 使用配置中的dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 准备模型进行LoRA训练
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print("LoRA model loaded. Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    train_dataset = CoTDataset(
        processor=processor,
        cot_files=args.cot_files,
        base_dir=args.base_dir,
        max_samples=args.max_samples,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=TRAINING_CONFIG["weight_decay"],  # 使用配置中的权重衰减
        logging_steps=TRAINING_CONFIG["logging_steps"],  # 使用配置中的日志步数
        save_steps=TRAINING_CONFIG["save_steps"],  # 使用配置中的保存步数
        fp16=False,  # 禁用 FP16 避免梯度缩放问题
        bf16=False,  # 也禁用 BF16，使用 FP32 训练
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],  # 使用配置中的梯度裁剪阈值
        save_total_limit=2,  # 保留2个检查点
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,  # 启用梯度检查点节省显存
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],  # 使用配置中的梯度累积步数
        optim="adamw_torch",  # 使用更稳定的AdamW优化器
        remove_unused_columns=False,  # 保留所有列避免数据处理开销
        dataloader_pin_memory=False,  # 禁用 pin memory 节省显存
        warmup_steps=TRAINING_CONFIG["warmup_steps"],  # 使用配置中的预热步数
        lr_scheduler_type="linear",  # 使用线性调度器，更稳定
        dataloader_drop_last=False,  # 不丢弃最后一个不完整的batch
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 禁用重入式梯度检查点
        max_steps=TRAINING_CONFIG["max_steps"],  # 使用配置中的最大步数
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[GradientNaNCallback()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
