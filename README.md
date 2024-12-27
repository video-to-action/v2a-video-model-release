# Video to Actions -- Video Models
<!-- 
``
This is the official 
`` -->

### Grounding Video Models to Actions through Goal Conditioned Exploration
[[Project page]](https://video-to-action.github.io/)
[[Paper]](https://arxiv.org/pdf/2411.07223)
[[ArXiv]](https://arxiv.org/abs/2411.07223)

[Yunhao Luo](https://devinluo27.github.io/)<sup>1,2</sup>,
[Yilun Du](https://yilundu.github.io/)<sup>3</sup>

<sup>1</sup>Georgia Tech,
<sup>2</sup>Brown,
<sup>3</sup>Harvard

This codebase contains code to train the video model in "*Grounding Video Models to Actions through Goal Conditioned Exploration*". 
For experiments in the robotic environments, please see [video-to-action-release](https://github.com/video-to-action/video-to-action-release) repo.

<!-- repo.  -->
<!-- this [repo](https://github.com/video-to-action/video-to-action-release). -->


## 🛠️ Installation
The required conda environment is identical to the environment `v2a_libero_release` as described in the installation section of [video-to-action-release](https://github.com/video-to-action/video-to-action-release?tab=readme-ov-file#%EF%B8%8F-installation).
<!-- repo. Please checkout it out and install the environment. -->



## 🗃️ Download Libero Demonstrations
Please refer to Libero documentation [link](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html) for the dataset downloading. 
Specifically, this codebase by default uses data from `LIVING_ROOM_SCENE5` and `LIVING_ROOM_SCENE6` in `libero_100` dataset.  
We provide a data preprossing script to prepare the data to train video models. If you would like to train video models on any other scenes, you can first preprocess the downloaded `.hdf5` Libero data and modify the dataloader accordingly.


## 📦 Data Preprocessing
The downloaded Libero demonstrations are stored in `.hdf5` file. We provide a script `flowdiffusion/libero/lb_extract_imgs.py` to extract the images. The commands are shown below. 
To successfully extract the data, correct Libero data file paths should be used. Please check out the note inside the script.

<!-- before running and download the Libero data. -->

```bash
cd flowdiffusion
sh ./libero/lb_extract_imgs.sh
```

<!-- We extract the demonstractions  -->


## 🕹️ Train a Model
With the extracted image data (which by default are stored in `datasets/libero`), you can now start training the video model. 

Note that the training requires 4 GPUs, each with at least 22GB Memory.  

To launch the training of the video model for Libero, run
```bash
cd flowdiffusion
sh libero/train_libero.sh
```
You can change the `$config` variable inside the script above to launch different experiments. You can refer to the default config in the script as template. Checkpoints will be saved in `flowdiffusion/logs`.

A pre-trained video model is provided [here](https://github.com/video-to-action/video-to-action-release?tab=readme-ov-file#%EF%B8%8F-prepare-data).

<!-- 💭 -->
## 📊 Video Model Inference

We provide a script to sample from the video model given a path to an image file and a text condition. An example of a start observation image is in the `examples` folder.
```bash
cd flowdiffusion
sh libero/plan_libero.sh
```


## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.


## 🙏 Acknowledgement
* The implementation of this codebase is based on [AVDC](https://github.com/flow-diffusion/AVDC).

Contact [Yunhao Luo](https://devinluo27.github.io/) if you have any questions or suggestions.


## 📝 Citations
If you find our work useful, please consider citing:
<!-- TODO: -->
```
@misc{luo2024groundingvideomodelsactions,
      title={Grounding Video Models to Actions through Goal Conditioned Exploration}, 
      author={Yunhao Luo and Yilun Du},
      year={2024},
      eprint={2411.07223},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.07223}, 
}
```



<!-- 
To resume training, you can use `-c` `--checkpoint_num` argument.  
```bash
# This will resume training with 1st checkpoint (should be named as model-1.pt)
python train_mw.py --mode train -c 1
``` -->

<!-- 


