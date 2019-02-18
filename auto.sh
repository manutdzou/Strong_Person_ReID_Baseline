#!/bin/sh

# python train.py ./config/cuhk_softmax.yaml
# python train.py ./config/cuhk_softmax_triplet.yaml

# python train.py ./config/market_softmax.yaml
python train.py ./config/market_softmax_triplet.yaml

python train.py ./config/duke_softmax.yaml
python train.py ./config/duke_softmax_triplet.yaml

python train.py ./config/ntu_softmax.yaml
python train.py ./config/ntu_softmax_triplet.yaml

python train.py ./config/msmt_softmax.yaml
python train.py ./config/msmt_softmax_triplet.yaml
