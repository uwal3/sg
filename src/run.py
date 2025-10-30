import os
from argparse import ArgumentParser

import torch
from diffusers import DiffusionPipeline
from train.model import SkinDecoder
from data.template import NEGATIVE_PROMPT

parser = ArgumentParser()
parser.add_argument("--prompt", type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe.to(device)

decoder = SkinDecoder().to(device)
decoder.load_state_dict(torch.load("checkpoint/decoder.pt", weights_only=True))
decoder = decoder.to(torch.float16)
decoder.eval()

pipe.vae.decoder = decoder

prompts = [args.prompt]

images = pipe(
    prompts,
    num_inference_steps=20,
    negative_prompt=[NEGATIVE_PROMPT] * len(prompts),
).images

if not os.path.exists("output"):
    os.mkdir("output/")
for i, img in enumerate(images):
    img.save(f"output/{i}.png")
