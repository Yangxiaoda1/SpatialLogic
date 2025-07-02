import os
import openai
import base64
import json
import time
import re
from openai.error import APIError, Timeout
from PIL import Image
import io
openai.verify_ssl_certs=False
# OpenAI
openai.api_key = "sk-..."
openai.api_base = "https://。。。"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def compress_image_to_base64(image_path: str, max_width: int = 800, quality: int = 70) -> str:
    with Image.open(image_path) as img:
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

def call_with_retries(func, max_retries=5, initial_backoff=1.0, **kwargs):
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            return func(**kwargs)
        except (APIError, Timeout) as e:
            code = getattr(e, 'http_status', None)
            if attempt < max_retries and (isinstance(e, Timeout) or (code and 500 <= code < 600)):
                print(f"[Warning] 第{attempt}次失败: {e}, {backoff}s后重试...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"[Error] 第{attempt}次失败: {e}, 不再重试.")
                raise

def build_entry(img1_path, img2_path, rel_dir, img1_name, img2_name):
    b64_1 = compress_image_to_base64(img1_path)
    b64_2 = compress_image_to_base64(img2_path)
    task_name = os.path.basename(rel_dir)

    user_prompt = (
        "1. 请你观察两个图片中的内容，仔细对比两个状态的不同，"
        "提示：两个状态具有时序关系(不一定哪个先发生)\n"
        f"2. 图中机械臂在完成“{task_name}”任务, 请分析并判断图像img1和img2中哪个更接近任务完成。"
    )
    sys_msg = {
        "role": "system",
        "content": "You are a vision-enabled assistant. Respond with only valid JSON for the target field."
    }
    usr_msg = {
        "role": "user",
        "content": f"""{user_prompt}

img1:
![](data:image/jpeg;base64,{b64_1})

img2:
![](data:image/jpeg;base64,{b64_2})

请返回如下 JSON 结构：
{{
  "state_difference": {{"img1":"...","img2":"..."}},
  "analysis":"...",
  "closer_to_completion":"img1" 或 "img2"
}}"""
    }
    resp = call_with_retries(
        openai.ChatCompletion.create,
        model="gpt-4o-mini",
        messages=[sys_msg, usr_msg],
        temperature=0,
        max_tokens=500,
        request_timeout=60
    )
    content = resp.choices[0].message.content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'```\s*$', '', content).strip()
    try:
        target = json.loads(content)
    except json.JSONDecodeError:
        print(f"[Error] 解析 JSON 失败: {rel_dir},\n{content}\n")
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type":"image","image":f"{rel_dir}/{img1_name}","image_id":"img1"},
                    {"type":"image","image":f"{rel_dir}/{img2_name}","image_id":"img2"},
                    {"type":"text","text":user_prompt}
                ]
            }
        ],
        "target": target
    }

if __name__ == '__main__':
    clips_root = os.getcwd()
    results = []
    for root, dirs, files in os.walk(clips_root):
        images = sorted(
            [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))],
            key=natural_keys
        )
        n = len(images)
        if n < 3:
            continue 

        first_idx = 0
        mid_idx   = n // 2
        last_idx  = n - 1

        pairs = [
            (mid_idx, first_idx),   # 第一轮：中间帧 vs 第一帧
            (last_idx, mid_idx)     # 第二轮：最后一帧 vs 中间帧
        ]
        rel_dir = os.path.relpath(root, clips_root).replace(os.sep, '/')

        for i, j in pairs:
            img1_name, img2_name = images[i], images[j]
            img1_path = os.path.join(root, img1_name)
            img2_path = os.path.join(root, img2_name)
            print(f"Processing {rel_dir}: frames {i} -> {j}")
            entry = build_entry(img1_path, img2_path, rel_dir, img1_name, img2_name)
            if entry:
                results.append(entry)

    out_file = os.path.join(clips_root, 'all_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"输出已写入 {out_file}")

