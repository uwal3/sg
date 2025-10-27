import os
from template import NEGATIVE_PROMPT

import h5py
import torch
import pandas as pd
from diffusers import DiffusionPipeline
from tqdm import tqdm

if not os.path.exists("data/pairs"):
    os.mkdir("data/pairs")

batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe.to(device)

meta = pd.read_csv("data/meta.csv")

with h5py.File("data/latents.h5", "w") as ds:
    latents_ds = ds.create_dataset("latents", shape=(len(meta), 4, 64, 64))

    for i in tqdm(range(0, len(meta), batch_size), desc="Generating latents"):
        batch = meta.iloc[i : i + batch_size]
        prompts = [row["caption"] for _, row in batch.iterrows()]
        latents = (
            pipe(
                prompts,
                num_inference_steps=20,
                output_type="latent",
                negative_prompt=[NEGATIVE_PROMPT] * len(prompts),
            ).images.cpu()
            / 0.18215  # sd 1.5 unet scaling
        )

        latents_ds[i : i + len(batch)] = latents
