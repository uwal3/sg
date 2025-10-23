#!/bin/bash

curl -L -o /tmp/skins.zip \
  https://www.kaggle.com/api/v1/datasets/download/amiralimollaei/minecraft-skins-hq-classified

unzip /tmp/skins.zip -d /tmp/skins

mkdir -p ./data/skins

mv /tmp/skins/skins-classified/_good/good ./data/skins