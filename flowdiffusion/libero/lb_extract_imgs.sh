#!/bin/bash

source ~/.bashrc
source activate v2a_libero_release


{
## make sure the command line terminal is in the folder `flowdiffusion`

python libero/lb_extract_imgs.py
exit 0

}