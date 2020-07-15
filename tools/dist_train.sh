#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=${RANDOM+10000}

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --resume_from work_dirs/condinst_r50_caffe_fpn_gn_1x_4gpu/latest.pth --launcher pytorch ${@:3}
