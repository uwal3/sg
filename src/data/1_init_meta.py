import os
import uuid

import pandas as pd
from tqdm import tqdm

meta = pd.DataFrame(columns=["id", "path"])

for file in tqdm(os.listdir("data/skins")):
    if file.endswith(".png"):
        id = uuid.uuid4()
        new_name = f"{id}.png"
        os.rename(
            os.path.join("data/skins", file), os.path.join("data/skins", new_name)
        )
        meta = pd.concat(
            [
                meta,
                pd.DataFrame(
                    {"id": [id], "path": [os.path.join("data/skins", new_name)]}
                ),
            ],
            ignore_index=True,
        )

meta = meta.sample(50000)
meta.to_csv("data/meta.csv", index=False)
