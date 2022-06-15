#!/bin/bash
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export OMP_NUM_THREADS=1
# avoid NCCL errors, https://github.com/PyTorchLightning/pytorch-lightning/issues/4420
export LOAD=0
# launch your script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    train.py\
    $LOAD