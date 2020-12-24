#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=789456 --master_addr=127.0.0.4 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
