#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import torch
import json
import argparse
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import math


class VideoFinetuneDataset(Dataset):
    def __init__(self, processor, image_path, task_path, window_sizes=[5, 10, 15], max_length=128):
        self.processor = processor
        self.task_path = task_path
        self.window_sizes = window_sizes
        self.max_length = max_length
        self.samples = []
        self.task_desc = {}

        for task in os.listdir(task_path):
            taskname = task.split('_')[3].split('.')[0]
            self.task_desc[taskname] = {}
            with open(os.path.join(task_path, task), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for episode in data:
                    eid = str(episode['episode_id'])
                    self.task_desc[taskname][eid] = {}
                    for pri in episode['label_info']['action_config']:
                        for i in range(math.ceil(pri['start_frame']/10), math.floor(pri['end_frame']/10)):
                            self.task_desc[taskname][eid][i] = pri['action_text']

        for sub1 in os.listdir(image_path):
            for sub2 in os.listdir(os.path.join(image_path, sub1)):
                sub2path = os.path.join(image_path, sub1, sub2)
                num_frames = len(os.listdir(sub2path))
                for window_size in self.window_sizes:
                    for i in range(num_frames - window_size + 1):
                        if i not in self.task_desc.get(sub1, {}).get(sub2, {}):
                            continue
                        action = self.task_desc[sub1][sub2][i]
                        prompt = (f"一个人机器人在完成任务：{action}，请对比这两张图片哪个更接近任务完成。"
                                  "输出1表示第一张图片更接近任务完成，-1表示第二张图片更接近任务完成")
                        first_img = os.path.join(sub2path, f"{i:05d}.jpg")
                        last_img = os.path.join(sub2path, f"{i + window_size - 1:05d}.jpg")

                        for a, b, t in [(first_img, last_img, "-1"), (last_img, first_img, "1")]:
                            self.samples.append({
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": a},
                                        {"type": "image", "image": b},
                                        {"type": "text", "text": prompt}
                                    ]
                                }],
                                "target": t
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        target = sample["target"]
        image_inputs = [Image.open(c["image"]).convert("RGB") for m in messages for c in m["content"] if c["type"] == "image"]
        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = chat_prompt + "\nassistant\n" + target
        inputs = self.processor(text=full_text, image_inputs=image_inputs, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=self.max_length)
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        prefix_len = self.processor.tokenizer(chat_prompt + "\nassistant\n", return_tensors="pt").input_ids.shape[-1] - 1
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        if (labels != -100).sum().item() == 0:
            print("警告：当前样本 labels 全为 -100，跳过该样本！")

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class GradientNaNCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        for name, param in kwargs["model"].named_parameters():
            if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN 梯度出现在: {name}, grad max: {param.grad.max().item()}, min: {param.grad.min().item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全量微调 Qwen2.5-VL 完成任务评估")
    parser.add_argument("--image_path", type=str, default='/home/tione/notebook/SpacialLogic-Demo/clips', help="图片文件夹路径")
    parser.add_argument("--task_path", type=str, default='/home/tione/notebook/SpacialLogic-Demo/task', help="任务描述文件夹路径")
    parser.add_argument("--window_size", type=int, nargs='+', default=[5,10,15], help="滑动窗口大小列表，默认 [5,10,15]")
    parser.add_argument("--max_length", type=int, default=128, help="文本部分最大长度，默认128")
    parser.add_argument("--output_dir", type=str, default="/home/tione/notebook/SpacialLogic-Demo/qwenvl/full/mycheckpoint", help="输出目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数量")
    parser.add_argument("--batch_size", type=int, default=1, help="每个设备的 batch 大小（可根据显存调节）")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16
    )

    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_dataset = VideoFinetuneDataset(
        processor, args.image_path, args.task_path,
        window_sizes=args.window_size, max_length=args.max_length
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=2000,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        save_total_limit=4,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[GradientNaNCallback()]
    )

    trainer.train()
    trainer.save_model(args.output_dir)