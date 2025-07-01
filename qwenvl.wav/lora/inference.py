from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Qwen2.5-VL 模型与对应的 processor
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
model=base_model
# model=PeftModel.from_pretrained(
#     base_model,
#     "/home/tione/notebook/SpacialLogic-Demo/qwenvl/lora/mycheckpoint/checkpoint-25000",
#     torch_dtype="auto"
# )

model.eval()

min_pixels=1280*960
max_pixels=1280*960
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",min_pixels=min_pixels, max_pixels=max_pixels)

# 构造消息：输入两个图片，文本提示保持不变
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/tione/notebook/SpacialLogic-Demo/clips/327/648748/Retrieve corn from the shelf./00117.jpg",
                "image_id": "img1"
            },
            {
                "type": "image",
                "image": "/home/tione/notebook/SpacialLogic-Demo/clips/327/648748/Retrieve corn from the shelf./00104.jpg",
                "image_id": "img2"
            },
            {
                "type": "text",
                "text": "1.请你观察两个图片中的内容，仔细对比两个状态的不同，提示：两个状态具有时序关系(不一定哪个先发生)\n2.图中机械臂在完成“Retrieve corn from the shelf.”任务，请判断图像img1和img2中哪个更接近任务完成。"
            }
        ]
    }
]

# 使用 processor 生成 prompt 文本和多模态输入
chat_prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# 从消息中提取图像和视频输入（此处只有图片）
image_inputs, video_inputs = process_vision_info(messages)
# print('坚果墙1',image_inputs)
inputs = processor(
    text=[chat_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
# print('坚果墙2',inputs)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 使用模型生成输出，max_new_tokens 可根据需要调整
generated_ids = model.generate(**inputs, max_new_tokens=256)
# 对生成的 token 序列解码为文本（这里保留整个输出文本）
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("生成的文本：", generated_text)
# 如果需要仅提取 assistant 回答部分，可以如下处理：
assistant_response = generated_text.split("assistant")[-1].strip()

# print("哪个更接近任务完成", assistant_response)