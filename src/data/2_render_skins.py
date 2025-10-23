import os
import asyncio
import pandas as pd
from minepi import Skin
from PIL import Image
from tqdm import tqdm

meta = pd.read_csv("data/meta.csv")

meta["slim"] = False
meta["render_front"] = ""
meta["render_back"] = ""

os.mkdir("data/renders")


async def main():
    for i, row in tqdm(meta.iterrows(), total=len(meta)):
        img = Image.open(row["path"])
        skin = Skin(raw_skin=img)

        front_img: Image.Image = await skin.render_skin(vr=-25, hr=-35)
        back_img: Image.Image = await skin.render_skin(vr=-25, hr=145)

        front_path = f"data/renders/{row['id']}_front.png"
        back_path = f"data/renders/{row['id']}_back.png"

        front_img.save(front_path)
        back_img.save(back_path)

        meta.at[i, "slim"] = skin.is_slim
        meta.at[i, "render_front"] = front_path
        meta.at[i, "render_back"] = back_path


asyncio.run(main())

meta.to_csv("data/meta.csv", index=False)
