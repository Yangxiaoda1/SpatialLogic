import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import argparse
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from qwen_vl_utils import process_vision_info
from peft import prepare_model_for_kbit_training
import json
from peft import LoraConfig, get_peft_model

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
            taskpath = os.path.join(task_path, task)
            with open(taskpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for episode in data:
                    episodename = str(episode['episode_id'])
                    self.task_desc[taskname][episodename] = {}
                    prilist = episode['label_info']['action_config']
                    for pri in prilist:
                        for i in range(pri['start_frame'], pri['end_frame'] + 1):
                            self.task_desc[taskname][episodename][i] = pri['action_text']

        for sub1 in os.listdir(image_path):  # 327
            sub1path = os.path.join(image_path, sub1)
            for sub2 in os.listdir(sub1path):  # 648642
                sub2path = os.path.join(sub1path, sub2)
                num_frames = len(os.listdir(sub2path))
                for window_size in self.window_sizes:
                    for i in range(num_frames - window_size + 1):
                        if i not in self.task_desc.get(sub1, {}).get(sub2, {}):
                            continue
                        action = self.task_desc[sub1][sub2][i]
                        text_prompt = (
                            f"一个人机器人在完成任务：{action}，请对比这两张图片哪个更接近任务完成。"
                            "输出1表示第一张图片更接近任务完成，-1表示第二张图片更接近任务完成"
                        )
                        first_img = os.path.join(sub2path, f"{i:05d}.jpg")
                        last_img = os.path.join(sub2path, f"{i + window_size - 1:05d}.jpg")
                        sample1 = {
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": first_img},
                                    {"type": "image", "image": last_img},
                                    {"type": "text", "text": text_prompt}
                                ]
                            }],
                            "target": "-1"
                        }
                        sample2 = {
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": last_img},
                                    {"type": "image", "image": first_img},
                                    {"type": "text", "text": text_prompt}
                                ]
                            }],
                            "target": "1"
                        }
                        self.samples.append(sample1)
                        self.samples.append(sample2)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        target = sample["target"]
        # 生成聊天 prompt
        chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 拼接 prompt 和目标，格式：<prompt>\nassistant\n<target>
        full_text = chat_prompt + "\nassistant\n" + target

        # 提取图片信息（此处 image_inputs 是一个 PIL Image 对象列表）
        image_inputs, _ = process_vision_info(messages)

        # 使用统一编码，将文本和图像信息一起处理
        inputs = self.processor(
            text=[full_text],
            image_inputs=image_inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = inputs.input_ids.squeeze(0)         # [max_length]
        attention_mask = inputs.attention_mask.squeeze(0)   # [max_length]
        labels = input_ids.clone()  # 若需要屏蔽 prompt 部分损失，可将相应 token 置为 -100
        # from IPython import embed; embed()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    return batch_dict

# def find_lora_target_modules(model):
#     target_modules = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             target_modules.append(name)
#     return target_modules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LoRA 微调 Qwen2.5-VL 完成任务评估")
    parser.add_argument("--image_path", type=str, default='/home/tione/notebook/SpacialLogic-Demo/clips', help="图片文件夹路径")
    parser.add_argument("--task_path", type=str, default='/home/tione/notebook/SpacialLogic-Demo/task', help="任务描述文件夹路径")
    parser.add_argument("--window_size", type=list, default=[5,10,15], help="滑动窗口大小，默认5")
    parser.add_argument("--max_length", type=int, default=128, help="文本部分最大长度，默认128")
    parser.add_argument("--output_dir", type=str, default="/home/tione/notebook/SpacialLogic-Demo/qwenvl/mycheckpoint", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数量")
    parser.add_argument("--batch_size", type=int, default=128, help="每个设备的 batch 大小")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    args = parser.parse_args()

    # 加载 processor 与预训练模型
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto"
    )
    
    # print("#################################")
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name)
    
    
    # for name, module in model.named_modules():
    #     print(name)
    # 定义 LoRA 配置（可根据实际情况调整 r、lora_alpha、lora_dropout 及 target_modules）
    lora_config = LoraConfig(
        r=16,                # 低秩分解的秩
        lora_alpha=32,       # 缩放因子
        target_modules = [
            # 语言模型
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",

            # 视觉 transformer
            "qkv", "proj",

            # merger 模块
            "mlp.0", "mlp.2"
        ],
        lora_dropout=0.1,   # dropout 概率
        bias="none", #["q_proj", "v_proj","visual.merger.mlp.0","visual.blocks.31.mlp","visual.blocks.30.mlp"]
        task_type="CAUSAL_LM"  # 根据任务类型选择，例如因果语言建模
    )
    # 使用 LoRA 封装模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print("LoRA 模型参数概况：")#LoRA 模型参数概况： None
    model.print_trainable_parameters()

    # 构造 Dataset
    train_dataset = VideoFinetuneDataset(
        processor,
        image_path=args.image_path,
        task_path=args.task_path,
        window_sizes=args.window_size,
        max_length=args.max_length
    )

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        report_to="none"
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )

    # 开始 LoRA 微调训练
    trainer.train()
