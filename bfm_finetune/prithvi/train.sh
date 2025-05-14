#!/bin/bash

# make this script work from any cwd
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


torchrun \
        --standalone \
        --nproc_per_node=1 \
        --nnodes=1 \
        --rdzv_backend=c10d \
        $SCRIPT_DIR/finetune_species.py
