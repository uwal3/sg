import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch


batch_size = 8
model_id = "Qwen/Qwen3-VL-2B-Instruct"


meta = pd.read_csv("data/meta.csv")
meta["caption"] = ""

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2",
)


def make_chat(front: Image.Image, back: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": front},
                {"type": "image", "image": back},
                {
                    "type": "text",
                    "text": "Given the front and back images of this Minecraft skin, provide a detailed description of its appearance. Your description should include colors, patterns, and any notable features present on the skin. Use references to common themes, styles or characters if applicable.",
                },
            ],
        }
    ]
    return messages


for i in range(0, len(meta), batch_size):
    batch = meta.iloc[i : i + batch_size]
    renders = []

    for _, row in batch.iterrows():
        front_img = Image.open(row["render_front"]).convert("RGB")
        back_img = Image.open(row["render_back"]).convert("RGB")
        renders.append([front_img, back_img])

    chats = [make_chat(front, back) for front, back in renders]
    inputs = processor.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    print(inputs.input_ids.shape)
