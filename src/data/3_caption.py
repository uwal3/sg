from template import CAPTIONING_PROMPT

import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from tqdm import tqdm


batch_size = 16
model_id = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


meta = pd.read_csv("data/meta.csv")
meta["caption"] = ""

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    attn_implementation="flash_attention_2",
)
model.to(device)


def make_chat(front: Image.Image, back: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": front},
                {"type": "image", "image": back},
                {"type": "text", "text": CAPTIONING_PROMPT},
            ],
        }
    ]
    return messages


for i in tqdm(range(0, len(meta), batch_size)):
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
        padding=True,
        padding_side="left",
    )

    with torch.no_grad():
        outputs = model.generate(
            **{k: v.to(model.device) for k, v in inputs.items()},
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]
        captions = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    for (idx, _), caption in zip(batch.iterrows(), captions):
        meta.at[idx, "caption"] = caption

meta.to_csv("data/meta.csv", index=False)
