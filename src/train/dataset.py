import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image


class SgDataset(Dataset):

    def __init__(self, meta: pd.DataFrame, latents: h5py.Dataset):
        self.meta = meta
        self.latents = latents

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        image_path = self.meta.iloc[index]["path"]
        image = pil_to_tensor(Image.open(image_path).convert("RGBA"))
        image = image.float() / 255.0 * 2.0 - 1.0
        latent = torch.from_numpy(self.latents[self.meta.iloc[index]["original_index"]])

        return {"pixel_values": image, "latents": latent}
