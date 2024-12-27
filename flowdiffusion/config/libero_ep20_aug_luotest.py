import os.path as osp
from diffuser.utils import watch
#------------------------ base ------------------------#

## automatically make experiment names
## by labelling folders with these args
config_fn = osp.splitext(osp.basename(__file__))[0]
diffusion_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ##
    ('n_diffusion_steps', 'T'),
]
ds_name = 'libero_90_65To72'
base = {
    'dataset': ds_name,
    'diffusion': {
        'config_fn': '',
        
        'target_size': (128,128),
        'valid_n': 4,

        
        ## dataset
        'root_dir': '../datasets/libero/datasets_img/',
        'dset_name': ds_name,
        'use_episode_range': (0,20),
        'num_ep_per_task': 20,
        'last_frame_range_len': 10,
        'randomcrop': True,
        'dataset_config': dict(
            ## Aug
            rand_rot_degrees=(-15,15),
            rand_resize_ratio=(1.0,1.25),
            color_jitter=dict(
                brightness=0.3, contrast=0.3, hue=0.2, saturation=0.3,
            ),
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## -------------------
        ## training
        'sample_steps': 100,
        'train_num_steps': 180000,
        'train_batch_size': 2,


    },

    'plan': {
        'config_fn': '',
        
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'diffusion_epoch': 'latest', #

    },

}
