#!/bin/bash

source ~/.bashrc
source activate v2a_libero_release



## Put the downloaded pre-trained checkpoint to
## flowdiffusion/logs/libero-90-65To72/diffusion/libero_ep20_bs12_aug/

config="config/libero_ep20_bs12_aug.py"

## input obs image for the video model
img="../examples/lb_tk65_put_the_red_mug_on_the_left_plate_t0_ep49.png"
## text condition for the video model
# task="put the red mug on the left plate"
task="put the red mug on the right plate"

echo "$task"

{
# python train_thor.py --mode inference -c 30 -p $img -t "$task" -g 2 -n 100 # -c 24
CUDA_VISIBLE_DEVICES=${1:-0} \
python libero/train_libero.py --config ${config} --mode inference \
    --checkpoint_num 180000 --inference_path $img --text "$task" # -n 100

exit 0
}
