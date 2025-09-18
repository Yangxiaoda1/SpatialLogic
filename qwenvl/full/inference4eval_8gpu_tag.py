#!/usr/bin/env python
# coding=utf-8

import argparse
import csv
import os
import re
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import time
from concurrent.futures import ThreadPoolExecutor
import queue

# 设置多进程启动方法为spawn，解决CUDA多进程问题
mp.set_start_method('spawn', force=True)

# 设置PyTorch优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def load_model_on_gpu(gpu_id: int, model_name: str):
    """在指定GPU上加载模型"""
    torch.cuda.set_device(gpu_id)
    print(f"🔄 GPU {gpu_id}: 正在加载模型...")
    
    # 使用原始QwenVL模型路径加载processor（无词表扩展，直接使用原始processor）
    checkpoint_path = "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/ckpt/qwenvl/full/tag_only_good/checkpoint-24000"
    original_model_path = "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/ckpt/qwenvl/original/Qwen2.5-VL-7B-Instruct"
    
    processor = AutoProcessor.from_pretrained(original_model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        device_map=f"cuda:{gpu_id}",
        torch_dtype=torch.bfloat16,
        use_cache=True
    )
    model.eval()
    
    print(f"✅ GPU {gpu_id}: 模型和processor加载完成 (processor: 原始QwenVL, model: checkpoint)")
    return processor, model

def infer_on_gpu(processor, model, messages: list, max_new_tokens: int, num_beams: int, gpu_id: int):
    """在指定GPU上进行推理"""
    torch.cuda.set_device(gpu_id)
    
    with torch.inference_mode():
        # 调试信息：检查messages结构
        print(f"🔍 GPU {gpu_id}: 检查messages结构")
        for i, msg in enumerate(messages):
            print(f"  Message {i}: {msg}")
            if 'content' in msg:
                for j, content in enumerate(msg['content']):
                    print(f"    Content {j}: {content}")
        
        # 提取图像和视频输入
        image_inputs, video_inputs = process_vision_info(messages)
        print(f"🔍 GPU {gpu_id}: process_vision_info返回 - image_inputs: {type(image_inputs)}, video_inputs: {type(video_inputs)}")

        # 构建 chat prompt
        chat_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"🔍 GPU {gpu_id}: apply_chat_template返回 - chat_prompt: {type(chat_prompt)}, 长度: {len(chat_prompt) if chat_prompt else 'None'}")
        
        chat_prompt = chat_prompt + "\nassistant\n"

        # 构造 inputs - 添加详细调试信息
        print(f"🔍 GPU {gpu_id}: 准备调用processor")
        print(f"  - chat_prompt类型: {type(chat_prompt)}, 长度: {len(chat_prompt) if chat_prompt else 'None'}")
        print(f"  - image_inputs类型: {type(image_inputs)}, 长度: {len(image_inputs) if image_inputs else 'None'}")
        print(f"  - video_inputs类型: {type(video_inputs)}")
        
        # 检查chat_prompt中是否有None值
        if chat_prompt:
            for i, char in enumerate(chat_prompt):
                if char is None:
                    print(f"❌ GPU {gpu_id}: chat_prompt位置{i}发现None值")
                    break
        
        inputs = processor(
            text=[chat_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到指定GPU
        inputs = {k: v.to(f"cuda:{gpu_id}") for k, v in inputs.items()}

        # 生成
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=None
        )
        
        # 解码（保留特殊标记，避免<answer>标签被过滤）
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # 提取 assistant 部分
        answer = generated_text.split("assistant")[-1].strip()
        return answer

def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, model_name: str, max_new_tokens: int, num_beams: int, base_dir: str):
    """工作进程函数"""
    try:
        # 在指定GPU上加载模型
        processor, model = load_model_on_gpu(gpu_id, model_name)
        
        while True:
            try:
                # 从队列获取任务
                task = task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                
                row_idx, row = task
                
                # 处理路径，添加None值检查
                video_path1_raw = row.get("video_path1", "")
                video_path2_raw = row.get("video_path2", "")
                task_name = row.get("task_name", "")
                
                # 检查None值
                if video_path1_raw is None or video_path2_raw is None or task_name is None:
                    print(f"⚠️ GPU {gpu_id} 行{row_idx}: 发现None值 - video_path1: {video_path1_raw}, video_path2: {video_path2_raw}, task_name: {task_name}")
                    continue
                
                video_path1 = video_path1_raw.strip().replace('_/','./')
                video_path2 = video_path2_raw.strip().replace('_/','./')
                task_name = task_name.strip()
                
                # 拼接完整路径
                video_path1 = os.path.join(base_dir, video_path1)
                video_path2 = os.path.join(base_dir, video_path2)
                
                # 构建prompt（简化为只输出最终答案，不需要spatial details）
                prompt = (
                    f"你是一位空间感知专家。\n请仔细观察两张图片(img1和img2)。 提示: 这两张图片发生在同一个任务中，描述了不同的任务完成状态。\n任务名称: {task_name}\n请你直接给出最终答案。\n输出格式:\ncloser to completion: [img1/img2]"
                )

                # 调试：检查路径和prompt是否有None值
                print(f"🔍 GPU {gpu_id} 行{row_idx}: 检查数据")
                print(f"  - video_path1: {video_path1} (类型: {type(video_path1)})")
                print(f"  - video_path2: {video_path2} (类型: {type(video_path2)})")
                print(f"  - prompt: {prompt[:100]}... (类型: {type(prompt)}, 长度: {len(prompt) if prompt else 'None'})")
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": video_path1, "image_id": "img1"},
                        {"type": "image", "image": video_path2, "image_id": "img2"},
                        {"type": "text", "text": prompt}
                    ]
                }]

                # 推理
                start_time = time.time()
                answer_text = infer_on_gpu(
                    processor, model, messages, max_new_tokens, num_beams, gpu_id
                )
                inference_time = time.time() - start_time
                print(f"\n📝 GPU {gpu_id} 行{row_idx} 原始输出:\n{answer_text}\n")
                
                # 解析答案（统一格式：spatial details: ... 和 closer to completion: <answer>...</answer>）
                spatial_details_val = ""
                closer_val = ""
                img1_text = ""
                img2_text = ""

                text_raw = answer_text
                sd_tag = "spatial details:"
                cc_tag = "closer to completion:"

                # 解析 spatial details
                sd_idx = text_raw.find(sd_tag)
                if sd_idx != -1:
                    sd_start = sd_idx + len(sd_tag)
                    cc_idx = text_raw.find(cc_tag, sd_start)
                    if cc_idx != -1:
                        spatial_details_val = text_raw[sd_start:cc_idx].strip()
                    else:
                        spatial_details_val = text_raw[sd_start:].strip()

                # 解析 closer to completion（直接解析文本内容，不再有<answer>标签）
                cc_idx2 = text_raw.find(cc_tag)
                if cc_idx2 != -1:
                    cc_start = cc_idx2 + len(cc_tag)
                    closer_val_raw = text_raw[cc_start:].strip()
                    
                    # 直接解析文本内容
                    if closer_val_raw:
                        lower = closer_val_raw.lower()
                        if "img1" in lower:
                            closer_val = "img1"
                        elif "img2" in lower:
                            closer_val = "img2"
                        else:
                            closer_val = closer_val_raw

                # 回退解析（正则，兼容多行、大小写）
                if not spatial_details_val or not closer_val:
                    norm = text_raw
                    m_sd = re.search(r"spatial\s*details\s*:\s*(.*?)\s*(?:closer\s+to\s+completion\s*:|$)", norm, flags=re.IGNORECASE|re.DOTALL)
                    if not spatial_details_val and m_sd:
                        spatial_details_val = m_sd.group(1).strip()
                    m_cc = re.search(r"closer\s+to\s+completion\s*:\s*(.*)$", norm, flags=re.IGNORECASE|re.DOTALL)
                    if not closer_val and m_cc:
                        closer_val_raw = m_cc.group(1).strip()
                        # 直接解析文本内容（不再有<answer>标签）
                        if closer_val_raw:
                            low = closer_val_raw.lower()
                            if "img1" in low:
                                closer_val = "img1"
                            elif "img2" in low:
                                closer_val = "img2"
                            else:
                                closer_val = closer_val_raw

                # 从spatial_details中进一步拆分img1/img2段
                if spatial_details_val:
                    m_pair = re.search(r"img1\s*:\s*(.*?)\n\s*img2\s*:\s*(.*)", spatial_details_val, flags=re.IGNORECASE|re.DOTALL)
                    if m_pair:
                        img1_text = m_pair.group(1).strip()
                        img2_text = m_pair.group(2).strip()
                    else:
                        # 尝试仅有img1或img2的单段形式
                        m1 = re.search(r"img1\s*:\s*(.*)$", spatial_details_val, flags=re.IGNORECASE|re.DOTALL)
                        m2 = re.search(r"img2\s*:\s*(.*)$", spatial_details_val, flags=re.IGNORECASE|re.DOTALL)
                        if m1:
                            img1_text = m1.group(1).strip()
                        if m2:
                            img2_text = m2.group(1).strip()

                result_item = {
                    "row_idx": row_idx,
                    "video_path1": video_path1,
                    "video_path2": video_path2,
                    "task_name": task_name,
                    "spatial_details": spatial_details_val,
                    "target": closer_val,
                    "inference_time": inference_time,
                    "gpu_id": gpu_id
                }
                
                result_queue.put(result_item)
                
            except queue.Empty:
                continue
                
    except Exception as e:
        print(f"❌ GPU {gpu_id} 工作进程出错: {e}")
        result_queue.put({"error": f"GPU {gpu_id}: {e}"})

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 8卡并行推理")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gt_csv", type=str, default="/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/eval/gt.csv")
    parser.add_argument("--pred_csv", type=str, default="/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/eval/tagger_8gpu_24000.csv")
    parser.add_argument("--base_dir", type=str, default="/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/data/clips")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的GPU数量")
    args = parser.parse_args()

    # 检查GPU数量
    if torch.cuda.device_count() < args.num_gpus:
        print(f"⚠️ 可用GPU数量 ({torch.cuda.device_count()}) 少于请求数量 ({args.num_gpus})")
        args.num_gpus = torch.cuda.device_count()
    
    print(f"🚀 使用 {args.num_gpus} 张GPU进行并行推理")
    
    # 读取数据
    print("📊 正在读取数据...")
    with open(args.gt_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        raw_fieldnames = reader.fieldnames or []
        fieldnames = [name.strip().lstrip("\ufeff") for name in raw_fieldnames]
        reader.fieldnames = fieldnames
        required_cols = {"video_path1", "video_path2", "task_name", "spatial_details", "target"}
        missing = required_cols - set(fieldnames)
        if missing:
            raise ValueError(f"gt.csv 缺少必要列: {missing}")

        all_rows = list(reader)
        total_rows = len(all_rows)
        print(f"📝 总共 {total_rows} 行数据")

    # 创建任务队列和结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 填充任务队列
    for i, row in enumerate(all_rows):
        task_queue.put((i, row))
    
    # 添加结束信号
    for _ in range(args.num_gpus):
        task_queue.put(None)
    
    # 启动工作进程
    print("🔄 正在启动GPU工作进程...")
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, task_queue, result_queue, args.model_name, args.max_new_tokens, args.num_beams, args.base_dir)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    print("🚀 开始并行推理...")
    results = [None] * total_rows
    completed = 0
    
    with tqdm(total=total_rows, desc="推理进度") as pbar:
        while completed < total_rows:
            try:
                result = result_queue.get(timeout=1)
                if "error" in result:
                    print(f"❌ {result['error']}")
                    continue
                
                row_idx = result["row_idx"]
                results[row_idx] = result
                completed += 1
                pbar.update(1)
                
                # 显示进度
                if completed % 10 == 0:
                    avg_time = sum(r["inference_time"] for r in results if r is not None) / completed
                    pbar.set_postfix({
                        "完成": f"{completed}/{total_rows}",
                        "平均时间": f"{avg_time:.2f}s",
                        "GPU": f"{result['gpu_id']}"
                    })
                    
            except queue.Empty:
                continue
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 保存结果
    print("💾 正在保存结果...")
    with open(args.pred_csv, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["video_path1", "video_path2", "task_name", "spatial_details", "target"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            if item:
                writer.writerow({
                    "video_path1": item["video_path1"],
                    "video_path2": item["video_path2"],
                    "task_name": item["task_name"],
                    "spatial_details": item["spatial_details"],
                    "target": item["target"]
                })
    
    # 统计信息
    total_time = sum(r["inference_time"] for r in results if r is not None)
    avg_time = total_time / total_rows
    
    print(f"\n🎉 推理完成!")
    print(f"📊 总样本数: {total_rows}")
    print(f"⏱️ 总推理时间: {total_time:.2f}s")
    print(f"⚡ 平均推理时间: {avg_time:.2f}s/样本")
    print(f"🚀 8卡加速比: {40/avg_time:.1f}x")  # 假设原来40s/样本
    print(f"💾 结果保存到: {args.pred_csv}")

if __name__ == "__main__":
    main()
