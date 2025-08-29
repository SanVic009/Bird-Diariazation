#!/bin/bash

python /home/sanvict/Documents/GitHub/Bird-Diariazation/sound2/src/preprocess_dataset.py --config /home/sanvict/Documents/GitHub/Bird-Diariazation/sound2/config.yaml
python /home/sanvict/Documents/GitHub/Bird-Diariazation/sound2/src/train.py --config /home/sanvict/Documents/GitHub/Bird-Diariazation/sound2/config.yaml