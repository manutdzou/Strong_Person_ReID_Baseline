#!/bin/sh

python test.py ./config/cuhk_softmax.yaml

python test.py ./config/market_softmax.yaml

python test.py ./config/duke_softmax.yaml

python test.py ./config/ntu_softmax.yaml

# python train.py ./config/msmt_softmax.yaml

# python train.py ./config/msmt_softmax_triplet.yaml
