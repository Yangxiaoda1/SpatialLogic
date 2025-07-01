#!/usr/bin/env python
# coding=utf-8

import argparse
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_model(model_name: str):
    """
    加载并返回处理器和模型
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/home/tione/notebook/SpacialLogic-Demo/qwenvl/full/mycheckpoint/checkpoint-215260",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    return processor, model


def build_chat_prompt(processor, messages: list):
    """
    使用聊天模板构建 prompt，并返回完整的生成前缀
    """
    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return chat_prompt + "\nassistant\n"


def infer(processor, model, messages: list, max_new_tokens: int, num_beams: int):
    """
    对给定多模态消息进行推理，返回 assistant 的回答
    """
    # 提取图像和视频输入
    image_inputs, video_inputs = process_vision_info(messages)

    # 构建 chat prompt
    chat_prompt = build_chat_prompt(processor, messages)

    # 构造 inputs
    inputs = processor(
        text=[chat_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    # 移动到模型设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True
    )
    # 解码
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    # 提取 assistant 部分
    answer = generated_text.split("assistant")[-1].strip()
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL 图像/视频对比任务推理示例"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--image1",
        type=str,
        default='/home/tione/notebook/SpacialLogic-Demo/clips/327/648642/00009.jpg',
        help="第一张图片路径"
    )
    parser.add_argument(
        "--image2",
        type=str,
        default='/home/tione/notebook/SpacialLogic-Demo/clips/327/648642/00014.jpg',
        help="第二张图片路径"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "一个人机器人在完成任务：Retrieve cucumber from the shelf，"
            "请对比这两张图片哪个更接近任务完成。"
            "输出1表示第一张更接近任务完成，-1表示第二张更接近任务完成"
        ),
        help="用户提示文本"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="生成的新 token 数量"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="beam search 的 beam 数量"
    )
    args = parser.parse_args()

    # 构造 messages 结构
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": args.image1},
            {"type": "image", "image": args.image2},
            {"type": "text",  "text": args.prompt}
        ]
    }]

    # 加载模型与处理器
    processor, model = load_model(args.model_name)

    # 执行推理
    answer = infer(
        processor,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams
    )

    print("推理结果：", answer)