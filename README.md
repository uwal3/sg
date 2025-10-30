# Custom Stable Diffusion Decoder for Minecraft Skin Generation

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?style=for-the-badge&logo=pytorch)
![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.35.2-yellow?style=for-the-badge)

This project focuses on training a custom decoder for the Stable Diffusion v1.5 text-to-image model, specifically tailored for generating Minecraft skins.

## Key Results

The model was trained for **19 epochs**, achieving a final **validation loss (L1) of 0.22**, and learning key structural features of the skins.

## Project Goal

The goal was to adapt the powerful Stable Diffusion model to a highly specialized domain. Instead of full fine-tuning of the entire U-Net, a lightweight custom decoder is trained, that acts as a drop-in replacement for the standard VAE decoder.

## Architecture and Methodology

### 1. Data Pipeline (`src/data`)

A data pipeline was designed to automate the creation of the training dataset. It consists of the following steps:

1.  **`0_load_dataset.sh`**: Downloads the initial dataset of UV-textures
2.  **`1_init_meta.py`**: Creates metadata for the dataset
3.  **`2_render_skins.py`**: Renders UV-textures as 3D models
4.  **`3_caption.py`**: Annotates the rendered images by generating descriptive text prompts using Qwen-VL
5.  **`4_generate_latents.py`**: Calculates the latent vectors by feeding the generated prompts into the frozen U-Net of Stable Diffusion v1.5

### 2. Decoder Architecture (`src/train/model.py`)

The architecture is built using blocks from the `diffusers` library.

### 3. Training (`src/train/train.py`)

The model was trained on a dataset of **64,000+ samples** on an NVIDIA RTX 3090 Ti.

*   **Optimizer:** AdamW
*   **Loss Function:** L1
*   **Learning Rate:** `CosineAnnealingLR` with an initial value of 5e-4

## Limitations & Next steps

Key areas for enhancement:

* Texture detail: To move beyond the current blurry reconstructions, the next step is to use a perceptual loss
* Alpha channel artifacts: These will be resolved by implementing a masked loss function