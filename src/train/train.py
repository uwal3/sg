from model import SkinDecoder
from dataset import SgDataset

import torch
import h5py
import wandb
import pandas as pd
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.nn import L1Loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

config = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 5e-4,
    "val_split": 0.02,
    "random_state": 52,
    "model_architecture": "SkinDecoder_v1",
}

wandb.init(project="minecraft-skin-decoder", config=config)

full_meta_df = pd.read_csv("data/meta.csv")
latents_file = h5py.File("data/latents.h5", "r")
latents = latents_file["latents"]

if "original_index" not in full_meta_df.columns:
    full_meta_df["original_index"] = full_meta_df.index

full_meta_df = full_meta_df[full_meta_df["slim"] == False]

train_meta_df, val_meta_df = train_test_split(
    full_meta_df, test_size=0.02, random_state=52
)

train_meta_df.reset_index(drop=True, inplace=True)
val_meta_df.reset_index(drop=True, inplace=True)


train_dataset = SgDataset(meta=train_meta_df, latents=latents)
val_dataset = SgDataset(meta=val_meta_df, latents=latents)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
fixed_val_batch = next(iter(val_dataloader))
fixed_latents = fixed_val_batch["latents"].to(device)
fixed_images_normalized = (fixed_val_batch["pixel_values"] + 1) / 2


# ======= TRAIN =======

model = SkinDecoder().to(device)
optimizer = AdamW(model.parameters(), lr=config["lr"])
loss_fn = L1Loss()
scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

best_val_loss = float("inf")

for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0.0
    for batch in tqdm(
        train_dataloader, desc=f"epoch {epoch+1}/{config['epochs']} [train]"
    ):
        optimizer.zero_grad()

        latents_batch = batch["latents"].to(device)
        images_batch = batch["pixel_values"].to(device)

        reconstructed_images = model(latents_batch)["sample"]
        loss = loss_fn(reconstructed_images, images_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        reconstructed_samples = model(fixed_latents)["sample"].cpu()
        reconstructed_samples = (reconstructed_samples + 1) * 127.5
        grid = make_grid(
            torch.cat([fixed_images_normalized, reconstructed_samples]),
            nrow=config["batch_size"],
        )
        wandb_image = wandb.Image(grid, caption=f"epoch {epoch+1}")

        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
            latents_batch = batch["latents"].to(device)
            images_batch = batch["pixel_values"].to(device)

            reconstructed_images = model(latents_batch)["sample"]
            loss = loss_fn(reconstructed_images, images_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    scheduler.step()

    wandb.log(
        {
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "reconstructions": wandb_image,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch + 1,
        }
    )
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"best_decoder.pt")

# =====================

wandb.finish()
latents_file.close()
