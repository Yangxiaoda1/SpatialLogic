from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Qwen2.5-VL 模型与对应的 processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 构造消息：输入两个图片，文本提示保持不变
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/data/home/yangxiaoda/SpacialVLM/2.png"},
            {"type": "image", "image": "/data/home/yangxiaoda/SpacialVLM/3.png"},
            {"type": "text", "text": "一个人机器人在完成任务：把黄瓜放到手推车的塑料袋里，请对比这两张图片哪个更接近任务完成。输出1表示第一张好更接近任务完成，-1表示第二张好更接近任务完成"}
        ]
    }
]

# 使用 processor 生成 prompt 文本和多模态输入
chat_prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# 从消息中提取图像和视频输入（此处只有图片）
image_inputs, video_inputs = process_vision_info(messages)
print('坚果墙1',image_inputs)
inputs = processor(
    text=[chat_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
print('坚果墙2',inputs)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 使用模型生成输出，max_new_tokens 可根据需要调整
generated_ids = model.generate(**inputs, max_new_tokens=128)
# 对生成的 token 序列解码为文本（这里保留整个输出文本）
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 如果需要仅提取 assistant 回答部分，可以如下处理：
assistant_response = generated_text.split("assistant")[-1].strip()

print("哪个更接近任务完成", assistant_response)