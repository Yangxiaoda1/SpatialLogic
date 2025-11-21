#!/usr/bin/env python
# coding=utf-8
"""
Train Qwen2.5-VL on CoT-style JSON data with lazy loading.
延迟加载版本 - 图片按需加载，适合大规模数据集

特性：
- 延迟加载：启动快速，图片按需加载
- LRU缓存：自动管理内存中的图片缓存
- 分布式训练：支持多GPU/多节点训练
- 优化性能：针对固定数据格式优化

使用方法：
  python train_cot_pdsh.py --cot_files data.json
"""

# ==================== 默认配置区域 ====================
# 这些是默认值，可以通过命令行参数覆盖
DEFAULT_CONFIG = {
    "base_dir": "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/data/clips",  # 项目根目录
    "model_path": "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/ckpt/qwenvl/original/Qwen2.5-VL-7B-Instruct",  # 模型路径
    "output_dir": "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/ckpt/qwenvl/full/cot_only",  # 输出目录
    "log_dir": "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/log/cot_only",  # TensorBoard 日志目录
    "cot_files": [],  # COT数据文件列表，用户必须通过命令行参数指定
    "epochs": 2,                    # 训练轮数
    "batch_size": 1,                # 批次大小（保持为1）
    "learning_rate": 2e-7,          # 学习率（进一步降低到1e-7，避免梯度爆炸）
    "max_samples": 0,             # 最大样本数（0表示使用全部数据）
    "max_steps": 0,                 # 最大训练步数（设为0表示根据数据量自动计算）
    "gradient_accumulation_steps": 2,   # 梯度累积步数（减少到2，最大程度节省显存）
    "warmup_steps": 1000,            # 预热步数
    "save_steps": 3000,              # 保存检查点步数
    "logging_steps": 10,            # 日志记录步数（减少控制台输出频率）
    "max_grad_norm": 0.3,           # 梯度裁剪阈值（降低到0.3，更保守的梯度控制）
    "weight_decay": 0.01,           # 权重衰减（全量微调使用更大的权重衰减）
    "max_target_length": 1500,      # 目标文本最大长度（减少到2000进一步节省显存）
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
import time
from typing import List, Dict, Any
from tqdm import tqdm
from collections import OrderedDict

# 可选快速JSON解析
try:
    import orjson as _orjson
except Exception:
    _orjson = None

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
# TensorBoard 导入，兼容不同环境
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        print("Warning: tensorboard not available, TensorBoard logging will be disabled")

# 导入accelerate相关模块
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not available, falling back to basic training")


# 延迟加载数据集 - 图片按需加载，避免预处理


# 已移除 LRU 缓存逻辑


class LazyCoTDataset(Dataset):
    """延迟加载版本的 CoT 数据集，避免启动时预处理所有图片"""
    
    def __init__(self, processor, cot_files: List[str], base_dir: str, max_samples: int = None, 
                 max_target_length: int = 3000):
        self.processor = processor
        self.max_target_length = max_target_length
        self.base_dir = base_dir
        self.max_samples = max_samples  # 保存原始值
        
        # 已移除图片缓存逻辑：始终按需读取图片
        self.image_cache = None
        
        # 仅加载元数据，不预处理图片
        self.samples_metadata = []
        
        print(f"🚀 延迟加载模式：仅加载样本元数据...")
        start_time = time.time()
        file_names = [os.path.basename(f) for f in cot_files]
        print(f"📁 计划加载 {len(cot_files)} 个文件: {file_names}")
        print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        

        for jf in cot_files:
            jf_abs = jf if os.path.isabs(jf) else os.path.abspath(jf)
            if not os.path.exists(jf_abs):
                print(f"⚠️ [WARN] CoT file not found: {jf_abs}")
                continue
                
            # 加载JSON文件
            with open(jf_abs, 'rb') as f:
                if _orjson is not None:
                    data = _orjson.loads(f.read())
                else:
                    import json
                    data = json.load(f)
                    print("加载json文件完成", flush=True)

            total_items = len(data) if isinstance(data, list) else 0
            print(f"📦 文件: {os.path.basename(jf_abs)} | 样本数: {total_items}", flush=True)

            added_samples = 0

            iterator = data
            try:
                iterator = tqdm(data, desc=f"处理 {os.path.basename(jf_abs)}", unit="样本")
            except Exception:
                pass

            for item in iterator:
                messages = item.get('messages', [])
                target = item.get('target', {})
                
                # 固定格式：第一个消息的前两个content是图片
                content = messages[0]['content']
                img1_path = content[0]['image'].replace('_/','./')
                img2_path = content[1]['image'].replace('_/','./')
                
                # 处理相对路径
                if not os.path.isabs(img1_path):
                    img1_path = os.path.join(base_dir, img1_path)
                    content[0]['image'] = img1_path
                if not os.path.isabs(img2_path):
                    img2_path = os.path.join(base_dir, img2_path)
                    content[1]['image'] = img2_path
                
                image_paths = [img1_path, img2_path]

                closer = target.get("closer to completion", "")
                spatial_details = target.get("spatial details") 
                target_text = f"spatial details: {spatial_details}\ncloser to completion: {closer}"
                
                metadata = {
                    'messages': messages,
                    'target_text': target_text,
                    'image_paths': image_paths,  # 预处理的图片路径列表
                }
                
                self.samples_metadata.append(metadata)
                added_samples += 1
                
                # 达到最大样本数时停止
                if self.max_samples and len(self.samples_metadata) >= self.max_samples:
                    break
            
            if self.max_samples and len(self.samples_metadata) >= self.max_samples:
                break

            print(
                f"✅ 完成 {os.path.basename(jf_abs)} | 元数据成功: {added_samples}",
                flush=True,
            )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"✅ 元数据加载完成: {len(self.samples_metadata)} 个样本")
        print(f"🔧 延迟加载模式：图片按需加载，无缓存")
        print(f"⏰ 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    def __len__(self):
        return len(self.samples_metadata)
    

    
    def _load_image_cached(self, image_path: str):
        """图片加载"""
        try:
            img = Image.open(image_path)
            # 确保图片是RGB格式，处理各种颜色模式
            if img.mode != 'RGB':
                if img.mode in ('RGBA', 'LA'):
                    # 有透明通道的图片，先转换为白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    else:  # LA mode
                        background.paste(img.convert('RGB'))
                    img = background
            else:
                    # 其他模式直接转换
                    img = img.convert('RGB')
            
            return img
        except Exception as e:
            return None

    def __getitem__(self, idx):
        metadata = self.samples_metadata[idx]
        
        # 使用预处理的目标文本
        target_text = metadata['target_text']
        
        # 限制目标文本长度
        if len(target_text) > self.max_target_length:
            target_text = target_text[:self.max_target_length]
        
        # 使用预处理的图片路径快速加载
        image_inputs = []
        for image_path in metadata['image_paths']:
            img = self._load_image_cached(image_path)
            if img is None:
                # 图片加载失败，跳过该样本
                return None
            else:
                image_inputs.append(img)
        
        # 处理对话
        messages = metadata['messages']
        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 与 add_generation_prompt=True 保持一致：不再手动追加 "\nassistant\n"
        full_text = chat_prompt + target_text
        
        inputs = self.processor(
            text=full_text, 
            images=image_inputs, 
            return_tensors='pt', 
            padding=False,
            truncation=False
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # 计算标签：前缀长度即模板产生的 chat_prompt 的 token 数
        prefix_len = self.processor.tokenizer(chat_prompt, return_tensors='pt').input_ids.shape[-1]
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        
        # 确保有效标签
        valid_label_count = (labels != -100).sum().item()
        if valid_label_count == 0:
            labels[-3:] = input_ids[-3:]
        
        if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # 收集视觉相关张量（如pixel_values、image_grid_thw等），并去除批维度
        extra = {}
        for k, v in inputs.items():
            if k not in ["input_ids", "attention_mask"] and torch.is_tensor(v):
                if v.dim() > 0 and v.size(0) == 1:
                    extra[k] = v.squeeze(0).cpu()
                else:
                    extra[k] = v.cpu()
        # 处理 image_grid_thw
        if 'image_grid_thw' in inputs and 'image_grid_thw' not in extra:
            grid = inputs['image_grid_thw']
            # 根据测试结果，processor返回的格式固定为 (n, 3)，直接使用
            if not torch.is_tensor(grid):
                grid = torch.tensor(grid, dtype=torch.long)
            extra['image_grid_thw'] = grid.cpu()
        
        # 确保张量在CPU上，让Trainer和accelerate处理设备分配
        sample = {
            "input_ids": input_ids.cpu(), 
            "attention_mask": attention_mask.cpu(), 
            "labels": labels.cpu()
        }
        sample.update(extra)
        return sample
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'total_samples': len(self.samples_metadata),
            'max_samples_limit': self.max_samples,
            'loading_mode': 'lazy_loading',
            'cache_enabled': False
        }


def create_collate_fn(processor):
    """
    创建处理变长序列的collate函数
    支持不同长度的input_ids、attention_mask和labels

    设计说明：
    1. 使用工厂函数模式，可以访问processor的tokenizer信息
    2. input_ids使用正确的pad_token_id填充，而不是硬编码的0
    3. 统一处理padding逻辑，避免在__getitem__中重复padding
    4. 过滤掉None值（当图片加载失败且配置为跳过时）
    """
    # 记忆最近一次可用样本，用于在整批无效时兜底
    last_good_sample = {"value": None}

    def collate_fn(batch):
        # 过滤掉None值（图片加载失败的样本）
        valid_batch = [b for b in batch if b is not None]
        if not valid_batch:
            if last_good_sample["value"] is not None:
                # 使用最近一次有效样本兜底，避免训练中断
                print("⚠️  Batch contains only invalid samples, reusing last valid sample as fallback", flush=True)
                valid_batch = [last_good_sample["value"]]
            else:
                # 首批就无有效样本，只返回空张量以跳过该步（保证形状合法）
                pad_token_id = processor.tokenizer.pad_token_id or 0
                empty_ids = torch.tensor([pad_token_id])
                empty_mask = torch.tensor([1])
                empty_labels = torch.tensor([-100])
                return {
                    "input_ids": empty_ids.unsqueeze(0),
                    "attention_mask": empty_mask.unsqueeze(0),
                    "labels": empty_labels.unsqueeze(0),
                }

        # 获取所有样本的键
        keys = valid_batch[0].keys()

        # 找到最大长度
        max_length = max(len(b['input_ids']) for b in valid_batch)

        # 初始化结果字典
        result = {}

        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # 对于需要padding的张量
                padded_tensors = []
                for b in valid_batch:
                    tensor = b[key]
                    current_length = len(tensor)

                    if current_length < max_length:
                        # 需要padding
                        if key == 'labels':
                            # labels用-100填充
                            padding = torch.full((max_length - current_length,), -100, dtype=tensor.dtype)
                        elif key == 'input_ids':
                            # input_ids用pad_token_id填充
                            pad_token_id = processor.tokenizer.pad_token_id or 0
                            padding = torch.full((max_length - current_length,), pad_token_id, dtype=tensor.dtype)
                        else:
                            # attention_mask用0填充
                            padding = torch.zeros(max_length - current_length, dtype=tensor.dtype)

                        padded_tensor = torch.cat([tensor, padding])
                    else:
                        padded_tensor = tensor

                    padded_tensors.append(padded_tensor)

                # 堆叠所有padded张量
                result[key] = torch.stack(padded_tensors)
            else:
                # 视觉或其他张量
                if key == 'pixel_values':
                    # 将每个样本中的图像维拼接到一起，形成 (sum_images, C, H, W)
                    result[key] = torch.cat([b[key] if b[key].dim() >= 3 else b[key].unsqueeze(0) for b in valid_batch], dim=0)
                elif key == 'image_grid_thw':
                    # 统一将每个样本的 grid 变成 (n, 3)，然后在样本维拼接 -> (sum_images, 3)
                    fixed = []
                    for b in valid_batch:
                        g = b[key]
                        if torch.is_tensor(g):
                            if g.dim() == 1:
                                if g.numel() == 2:
                                    g = torch.tensor([1, int(g[0]), int(g[1])], dtype=torch.long).view(1, 3)
                                elif g.numel() == 3:
                                    g = g.view(1, 3).long()
                                else:
                                    raise ValueError(f"Unexpected image_grid_thw shape: {g.shape}")
                            elif g.dim() >= 2:
                                if g.size(-1) == 2:
                                    ones = torch.ones(*g.shape[:-1], 1, dtype=g.dtype)
                                    g = torch.cat([ones, g], dim=-1)
                                # 将可能存在的更高维前缀展平成 (n, 3)
                                g = g.view(-1, 3).long()
                        else:
                            # list/tuple -> 张量
                            if isinstance(g, (list, tuple)):
                                def to_thw_list(x):
                                    if isinstance(x, (list, tuple)):
                                        if len(x) == 2 and all(isinstance(n, (int, float)) for n in x):
                                            return [1, int(x[0]), int(x[1])]
                                        if len(x) == 3 and all(isinstance(n, (int, float)) for n in x):
                                            return [int(x[0]), int(x[1]), int(x[2])]
                                        return [to_thw_list(e) for e in x]
                                    return x
                                g_list = to_thw_list(g)
                                g = torch.tensor(g_list, dtype=torch.long)
                                g = g.view(-1, 3)
                            else:
                                raise TypeError("image_grid_thw must be Tensor or list/tuple")
                        fixed.append(g)
                    result[key] = torch.cat(fixed, dim=0)
                else:
                    # 其他定长张量，直接堆叠
                    result[key] = torch.stack([b[key] for b in valid_batch])

        # 更新最近一次可用样本
        try:
            last_good_sample["value"] = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in valid_batch[-1].items()
            }
        except Exception:
            pass

        return result

    return collate_fn


 

class TensorBoardCallback(TrainerCallback):
    """TensorBoard 回调，记录详细的训练信息"""
    
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or "/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/log"
        self.writer = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时初始化 TensorBoard writer"""
        if state.is_local_process_zero and TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(self.log_dir)
                print(f"📊 TensorBoard 日志将保存到: {self.log_dir}")
            except Exception as e:
                print(f"⚠️  TensorBoard 初始化失败: {e}")
                self.writer = None
    
    def on_step_end(self, args, state, control, **kwargs):
        """每步结束时记录训练信息"""
        if state.is_local_process_zero and self.writer is not None:
            try:
                # 记录学习率
                if hasattr(state, 'learning_rate'):
                    self.writer.add_scalar('train/learning_rate', state.learning_rate, state.global_step)
                
                # 记录训练损失
                if hasattr(state, 'log_history') and state.log_history:
                    latest_log = state.log_history[-1]
                    if 'loss' in latest_log:
                        self.writer.add_scalar('train/loss', latest_log['loss'], state.global_step)
            except Exception as e:
                print(f"⚠️  TensorBoard 记录失败: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时关闭 TensorBoard writer"""
        if state.is_local_process_zero and self.writer is not None:
            try:
                self.writer.close()
                print(f"📊 TensorBoard 日志已保存到: {self.log_dir}")
            except Exception as e:
                print(f"⚠️  TensorBoard 关闭失败: {e}")





class GradientNaNCallback(TrainerCallback):
    """
    梯度数值异常检测回调
    
    功能：
    1. 检测并修复NaN/Inf梯度，防止训练崩溃
    2. 不重复Trainer内置的梯度裁剪功能
    
    注意：
    - 梯度裁剪由TrainingArguments.max_grad_norm自动处理
    - 此回调在optimizer.step()之后执行，主要用于检测和日志记录
    - 对于真正的NaN/Inf梯度修复，建议在更早的阶段（如training_step中）处理
    """
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
            
        # 检查NaN和Inf梯度，只处理数值异常，梯度裁剪由Trainer内置功能处理
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad = p.grad.data
                if torch.isnan(grad).any():
                    print(f"⚠️  NaN gradient detected at {name}, replacing with zeros")
                    p.grad.data = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
                elif torch.isinf(grad).any():
                    print(f"⚠️  Inf gradient detected at {name}, replacing with zeros")
                    p.grad.data = torch.where(torch.isinf(grad), torch.zeros_like(grad), grad)



def main():
    parser = argparse.ArgumentParser(description="Full Fine-tune Qwen2.5-VL on CoT JSON")
    parser.add_argument("--cot_files", nargs="+", default=DEFAULT_CONFIG["cot_files"], 
                       help="COT数据文件列表 (支持多个文件，用空格分隔，如: cotdata/cot_50.0.json cotdata/cot_60.0.json)")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_CONFIG["base_dir"], help="Base dir to resolve relative image paths (should contain testset)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"], help="Local model dir")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help="Output dir")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_CONFIG["max_samples"], 
                       help="最大样本数限制 (设为0或负数表示使用全部数据，用于避免OOM)")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"], help="Maximum training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"], help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"], help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"], help="Save checkpoint steps")
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG["logging_steps"], help="Logging steps")
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"], help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"], help="Weight decay")
    parser.add_argument("--from_pretrained", action="store_true", help="Whether to load from a pretrained model")
    parser.add_argument("--accelerate", action="store_true", help="Whether to use accelerate for distributed training")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_target_length", type=int, default=3000,
                       help="Maximum target text length before truncation (default: 3000 for optimal memory usage)")
    parser.add_argument("--dataloader_workers", type=int, default=0,
                       help="Number of PyTorch DataLoader workers for image decoding (default: 0 to avoid multiprocessing issues)")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"],
                       help="Directory to save TensorBoard logs (default: from DEFAULT_CONFIG)")
    args = parser.parse_args()

    print(f"✅ 延迟加载模式（图片按需加载）")

    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化accelerator（如果可用且启用）
    if ACCELERATE_AVAILABLE and args.accelerate:
        accelerator = Accelerator()
        print(f"Using accelerate for distributed training")
        device = accelerator.device
        use_cuda = device.type == 'cuda'
    else:
        accelerator = None
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
    
    dtype = torch.float32  # 强制使用float32提高数值稳定性
    print(f"Using device: {device}, dtype: {dtype}")

    # 加载processor
    print(f"🔄 正在加载处理器: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    
    # 检查并设置pad_token_id，确保padding操作的安全性
    if processor.tokenizer.pad_token is None:
        print(f"⚠️  Warning: pad_token not set, using eos_token as pad_token")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"✅ pad_token_id: {processor.tokenizer.pad_token_id}")
    
    # 根据from_pretrained参数决定模型加载方式
    if args.from_pretrained:
        print(f"🔄 正在从预训练检查点加载模型: {args.model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            use_cache=False,  # 禁用 KV 缓存节省显存
            low_cpu_mem_usage=True,  # 减少 CPU 内存使用
        )
    else:
        print(f"🔄 正在从基础模型加载模型: {args.model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            use_cache=False,  # 禁用 KV 缓存节省显存
            low_cpu_mem_usage=True,  # 减少 CPU 内存使用
        )

    # 若词表有扩展，则调整嵌入矩阵大小
    try:
        model.resize_token_embeddings(len(processor.tokenizer))
        try:
            model.config.vocab_size = int(len(processor.tokenizer))
        except Exception:
            pass
        print(f"✅ 已根据 tokenizer 词表大小重置嵌入矩阵: {len(processor.tokenizer)}")
    except Exception as e:
        print(f"⚠️  重置嵌入矩阵失败: {e}")
    # 确保use_cache为False
    model.config.use_cache = False

    # 全量微调：设置所有参数为可训练
    for param in model.parameters():
        param.requires_grad = True
    
    print(f"全量微调模式：所有参数可训练")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 在 accelerator.prepare 之前启用梯度检查点
    if accelerator is not None:
        print(f"🔧 启用梯度检查点...")
        model.gradient_checkpointing_enable()
        print(f"✅ 梯度检查点已启用: {model.is_gradient_checkpointing}")
    
    # 使用accelerator准备模型（如果启用）
    if accelerator is not None:
        model = accelerator.prepare(model)
        print(f"Model prepared with accelerator")
    
    if use_cuda:
        print(f"模型加载完成. GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print(f"模型加载完成. 使用CPU训练")

    # 创建数据集
    print(f"🔄 创建数据集...")
    train_dataset = LazyCoTDataset(
        processor=processor,
        cot_files=args.cot_files,
        base_dir=args.base_dir,
        max_samples=args.max_samples,
        max_target_length=args.max_target_length,
    )

    # 计算实际的训练步数
    if args.max_steps <= 0:
        # 根据数据量和训练配置自动计算
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        if accelerator is not None:
            effective_batch_size *= accelerator.num_processes
        
        total_samples = len(train_dataset)
        steps_per_epoch = total_samples // effective_batch_size
        if total_samples % effective_batch_size != 0:
            steps_per_epoch += 1
        
        calculated_max_steps = steps_per_epoch * args.epochs
        print(f"📊 自动计算训练步数:")
        print(f"   - 总样本数: {total_samples}")
        print(f"   - 有效批次大小: {effective_batch_size}")
        print(f"   - 每轮步数: {steps_per_epoch}")
        print(f"   - 总训练步数: {calculated_max_steps}")
        
        # 更新warmup_steps，确保不超过总步数的50%
        if args.warmup_steps > calculated_max_steps * 0.5:
            args.warmup_steps = max(100, int(calculated_max_steps * 0.1))
            print(f"   - 调整预热步数: {args.warmup_steps}")
    else:
        calculated_max_steps = args.max_steps
        print(f"📊 使用指定的最大步数: {calculated_max_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,  # 使用args中的权重衰减
        logging_steps=args.logging_steps,  # 使用args中的日志步数
        save_steps=args.save_steps,  # 使用args中的保存步数
        fp16=False,  # 禁用 FP16
        bf16=True,  # 启用 BF16 混合精度训练，与 accelerate config 保持一致
        max_grad_norm=args.max_grad_norm,  # 使用args中的梯度裁剪阈值
        save_total_limit=5,  # 保留4个检查点
        logging_dir=args.log_dir,  # TensorBoard 日志目录
        report_to="tensorboard",  # 启用 TensorBoard
        dataloader_num_workers=0,  # 设置为0，避免多进程问题
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 使用args中的梯度累积步数
        optim="adamw_torch",  # 使用AdamW优化器，DeepSpeed兼容性更好
        remove_unused_columns=False,  # 保留所有列避免数据处理开销
        warmup_steps=args.warmup_steps,  # 使用args中的预热步数
        lr_scheduler_type="cosine",  # 使用余弦调度器，更适合全量微调
        dataloader_drop_last=False,  # 不丢弃最后一个不完整的batch
        max_steps=calculated_max_steps,  # 使用计算出的最大步数
        # 内存优化选项
        dataloader_pin_memory=False,  # 禁用pin_memory节省显存
        gradient_checkpointing=False,  # 已在代码中手动启用，避免 DDP 冲突
        # DeepSpeed 配置
        deepspeed=args.deepspeed,  # DeepSpeed 配置文件路径
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=create_collate_fn(processor),  # 使用工厂函数创建collate_fn
        callbacks=[TensorBoardCallback(log_dir=args.log_dir), GradientNaNCallback()],
    )



    print("🚀 开始训练模型...")
    print(f"训练配置: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"训练步数: {calculated_max_steps} steps, warmup_steps={args.warmup_steps}")
    
    # 训练模型
    trainer.train()
    
    print("✅ 训练完成！正在保存模型...")
    
    # 图片加载统计（延迟加载模式）
    print(f"📊 图片加载统计:")
    print(f"   - 延迟加载模式：图片按需加载，无缓存")
    print(f"   - 总样本数: {len(train_dataset)}")
    
    # 保存模型和processor
    if accelerator is not None:
        # 如果使用accelerator，等待所有进程完成
        accelerator.wait_for_everyone()
        # 只在主进程上保存模型和processor
        if accelerator.is_main_process:
            print(f"💾 正在保存模型到: {args.output_dir}")
            trainer.save_model(args.output_dir)
            
            print(f"💾 正在保存processor到: {args.output_dir}")
            try:
                processor.save_pretrained(args.output_dir)
                print(f"✅ Processor保存成功")
            except Exception as e:
                print(f"❌ Processor保存失败: {e}")
            
            # 验证关键文件是否存在
            required_files = ["preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"]
            for file in required_files:
                file_path = os.path.join(args.output_dir, file)
                if os.path.exists(file_path):
                    print(f"✅ {file} 存在")
                else:
                    print(f"❌ {file} 缺失")
            
            print(f"🎉 模型和processor已保存到: {args.output_dir}")
    else:
        print(f"💾 正在保存模型到: {args.output_dir}")
        trainer.save_model(args.output_dir)
        
        print(f"💾 正在保存processor到: {args.output_dir}")
        try:
            processor.save_pretrained(args.output_dir)
            print(f"✅ Processor保存成功")
        except Exception as e:
            print(f"❌ Processor保存失败: {e}")
        
        # 验证关键文件是否存在
        required_files = ["preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"]
        for file in required_files:
            file_path = os.path.join(args.output_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file} 存在")
            else:
                print(f"❌ {file} 缺失")
        
        print(f"🎉 模型和processor已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
