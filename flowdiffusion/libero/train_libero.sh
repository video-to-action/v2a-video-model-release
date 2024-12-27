#!/bin/bash

#SBATCH --time=70:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -J script-v2a-lb-video-model
#SBATCH -o ./trash/slurm/slurm-%j.out

source ~/.bashrc
source activate v2a_libero_release


## this config is just for testing, where batch size is 2 and only needs 1 GPU.
num_gpus=1
export config="config/libero_ep20_aug_luotest.py"

## this is the production config to use, which needs 4 GPUs
num_gpus=4
export config="config/libero_ep20_bs12_aug.py"


{

## The training requires 4 GPUs, each with at least 23GB Memory.
## the current work directory of the command line terminal should be flowdiffusion.

accelerate launch \
    --num_processes $num_gpus \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
\
libero/train_libero.py --config ${config} --mode train 

exit 0
}