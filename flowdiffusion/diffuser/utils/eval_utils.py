from colorama import Fore
import numpy as np
import torch, pdb, contextlib, os, json, imageio
import torch.nn.functional as F
from colorama import Fore
from torchvision import transforms as T
import os.path as osp

def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)


def save_imgs_1(imgs, full_paths):
    for i_m, img in enumerate(imgs):
        p_dir = osp.dirname(full_paths[i_m])
        os.makedirs(p_dir, exist_ok=True)
        imageio.imsave(full_paths[i_m], img)
    
    print(f'save {len(imgs)} pngs to:', p_dir)


def save_gif(imgs, root_dir, fname, dr=0.5, se_dr=1.5):
    '''
    dr: duration
    se_dr: duration for the start and end
    '''
    if type(imgs) == np.ndarray:
        assert imgs.dtype == np.uint8 and imgs.ndim == 4
    else:
        assert type(imgs) == list
        assert imgs[0].ndim == 3
        ## float 0-1 to uint8
        if imgs[0].dtype == np.float32:
            assert imgs[0].min() >= 0 and imgs[0].max() <= 1
            imgs = [(img * 255).astype(np.uint8) for img in imgs]
        else:
            assert imgs[0].dtype == np.uint8


    os.makedirs(root_dir, exist_ok=True)
    tmp = osp.join(root_dir, f'{fname}')
    ds = [dr,] * len(imgs)
    ds[0] = se_dr; ds[-1] = se_dr
    imageio.mimsave(tmp, imgs, duration=ds)
    print(f'[save_gif] to {tmp}')



def save_json(j_data: dict, full_path):
    with open(full_path, "w") as f:
        json.dump(j_data, f, indent=4)
    print(f'[save_json] {full_path}')

