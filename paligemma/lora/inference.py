import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# 模型与设备初始化
model_id = "google/paligemma-3b-pt-224"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
).to(device)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# 加载图片
img1_path = "/home/tione/notebook/SpacialLogic-Demo/clips/327/648748/Retrieve corn from the shelf./00104.jpg"
img2_path = "/home/tione/notebook/SpacialLogic-Demo/clips/327/648748/Retrieve corn from the shelf./00117.jpg"
image1 = Image.open(img1_path).convert("RGB")
image2 = Image.open(img2_path).convert("RGB")

# 构建 prompt（包含 image tokens + 明确任务定义）
prompt = (
    "<image> <image> The robot is performing the task: Retrieve corn from the shelf.\n"
    "Step 1: Describe what is happening in the first image.\n"
    "Step 2: Describe what is happening in the second image.\n"
    "Step 3: Based on these observations, which image is closer to completing the task? Why?\n"
    "Answer:"
)

inputs = processor(images=[image1, image2], text=prompt, return_tensors="pt").to(device, torch.bfloat16)

# 推理
outputs = model.generate(**inputs, max_new_tokens=300)
answer = processor.decode(outputs[0], skip_special_tokens=True).strip()

# 输出答案
print("Answer:", answer)