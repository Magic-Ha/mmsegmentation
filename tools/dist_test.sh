#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_addr=127.0.0.9 --master_port=29509 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

# python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.2 --master_port=29501 ./tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
