#!/bin/bash

mkdir tmp/

curl -L -o tmp/skins.zip \
  https://www.kaggle.com/api/v1/datasets/download/amiralimollaei/minecraft-skins-hq-classified

unzip tmp/skins.zip -d tmp/skins

rm -rf tmp/skins/skins-classified/bad_

rm tmp/skins.zip

mkdir -p data/skins

mv tmp/skins/skins-classified/good_/good/ data/skins

rm -frfr tmp/skins