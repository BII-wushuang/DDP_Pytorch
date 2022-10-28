#!/usr/bin/env bash
# Modify the GPUs you wish to use here
export CUDA_VISIBLE_DEVICES=1,2,3

# torch.distributed.launch arguments
n="${CUDA_VISIBLE_DEVICES//[^[:digit:]]/}"
GPUS=${#n}
PORT=${PORT:-28500}

# Variable number of arguments
ARGS=("$@")

# Add the current directory to ENV PYTHONPATH
PYTHONPATH="$(dirname $0)":$PYTHONPATH \

# Training script with arguments
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py $ARGS