#!/bin/bash
# TRAIN
torchrun \
        --nproc_per_node=1 \
        --nnodes=1 \
        --rdzv_backend=c10d \
        finetune_species.py
