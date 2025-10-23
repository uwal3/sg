import os
import uuid

import pandas as pd
from tqdm import tqdm

meta = pd.DataFrame(columns=["id", "path"])

for file in tqdm(os.listdir("data/skins/good")):
    if file.endswith(".png"):
        id = file[:-4]

        meta = pd.concat(
            [
                meta,
                pd.DataFrame(
                    {"id": [id], "path": [os.path.join("data/skins/good", file)]}
                ),
            ],
            ignore_index=True,
        )

meta = meta.sample(64000, random_state=52)
meta.to_csv("data/meta.csv", index=False)
