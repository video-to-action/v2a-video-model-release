from torch.utils.data import Dataset
from torchvideotransforms import video_transforms, volume_transforms
import random
from einops import rearrange
from glob import glob
from PIL import Image
import re, pdb
import numpy as np


class Sequential_Libero_Dataset_V2_Subgoal(Dataset):
    
    def __init__(self, path="../datasets/valid", 
                 dataset_name=None,
                 sample_per_seq=7, 
                 target_size=(128, 128), 
                 frameskip=None, 
                 randomcrop=False,
                 ## --- Luo ---
                 use_episode_range=(0,20),
                 num_ep_per_task=20,
                 ## define frames that can be the last frame in a data point
                 ## if 5, all the final 5 frames can be the last frame
                 last_frame_range_len:int=1,
                 dataset_config={},
                 ):
        '''
        Libero Dataset for Training the Video Model. Adapted From the AVDC Codebase.
        '''
        print("Preparing Sequential_Libero_Dataset_V2 dataset...")
        self.sample_per_seq = sample_per_seq
        self.target_size = target_size

        self.frame_skip = frameskip
        
        self.use_episode_range = use_episode_range
        self.num_ep_per_task = num_ep_per_task
        self.last_frame_range_len = last_frame_range_len
        assert 1 <= last_frame_range_len <= 10
        
        ## NEW
        self.subgoal_delta = dataset_config['subgoal_delta']
        self.subgoal_delta[1] = self.subgoal_delta[1] + 1
        assert self.frame_skip is None



        ## datasets/metaworld/metaworld_dataset/assembly/corner/000/00.png
        # sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        sequence_dirs, tk_names_input = self.glob_data_dirs(path, dataset_name)


        self.tasks = []
        self.sequences = []
        for i_s, seq_dir in enumerate(sequence_dirs):
            ## str task name
            task = seq_dir.split("/")[-4]
            ## int of the episode
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            # pdb.set_trace()
            tmp_fn_2 = lambda x: int(x.split("/")[-1].rstrip(".png"))
            seq = sorted(glob(f"{seq_dir}/agentview_rgb/*.png"), key=tmp_fn_2)
            self.sequences.append(seq)
            ## important: for the video model input
            # self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
            self.tasks.append( tk_names_input[i_s] )
    


        if randomcrop:
            ## Custom Aug, Note that the input image is (128,128), cannot dirctly crop
            self.transform = video_transforms.Compose([
                video_transforms.RandomRotation( degrees=dataset_config['rand_rot_degrees'] ), # (-15, 15)
                video_transforms.RandomResize( ratio=dataset_config['rand_resize_ratio'] ), # (1.0, 1.25)
                video_transforms.RandomCrop(size=target_size,),
                # brightness=0.3, contrast=0.3, hue=0.2, saturation=0.3
                video_transforms.ColorJitter(**dataset_config['color_jitter']),
                # video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
            
        else:
            assert 'rand_rot_degrees' not in dataset_config
            self.transform = video_transforms.Compose([
                ## no need
                # video_transforms.CenterCrop((128, 128)),
                # video_transforms.Resize(target_size),
                ## m (H x W x C) --> (C x m x H x W)
                volume_transforms.ClipToTensor(),
            ])
        print("Done")
        # pdb.set_trace()


    def get_samples(self, idx):
        seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None:

            if self.last_frame_range_len == 1:
                assert False
                ## AVDC default
                start_idx = random.randint(0, len(seq)-1)
                seq = seq[start_idx:]
                N = len(seq)
                samples = []
                for i in range(self.sample_per_seq-1):
                    samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
                samples.append(N-1)
            else:
                ## Luo's Impl
                ## two ends included
                last_idx = random.randint(len(seq)-self.last_frame_range_len, len(seq)-1)
                start_idx = random.randint(0, last_idx)
                sg_idxs = np.random.randint(low=self.subgoal_delta[0], 
                                high=self.subgoal_delta[1], size=self.sample_per_seq)

                # pdb.set_trace()
                sg_idxs = np.cumsum(sg_idxs)
                # pdb.set_trace()
                sg_idxs = sg_idxs + start_idx
                
                ## this will copy list, won't override
                # seq = seq[start_idx:(last_idx+1)]
                N = len(seq) - 1 # wrong: last_idx + 1
                samples = [start_idx,]
                for i_s in range(self.sample_per_seq-1):
                    samples.append(  min(sg_idxs[i_s], N) )
                # samples.append(N-1)
                # pdb.set_trace()
                # print(start_idx, last_idx,)
                # print(len(self.sequences[idx]), samples)

        else:
            assert False, 'should not use'
            start_idx = random.randint(0, len(seq)-1)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            samples = self.get_samples(idx)
            images = self.transform([Image.open(s) for s in samples]) # [c f h w]
            x_cond = images[:, 0] # first frame
            assert x_cond.shape == (3, *self.target_size)
            x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
            task = self.tasks[idx]
            return x, x_cond, task
        except Exception as e:
            print(e)
            return self.__getitem__(idx + 1 % self.__len__()) 
        


    def glob_data_dirs(self, path, dataset_name):
        '''
        analyze and get the dir path at episode level, 
        also return the truncated task name
        Returns:

        '''
        task_dirs = glob(f"{path}/{dataset_name}/*/", recursive=False)
        
        ## list of 'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate', ...
        task_dirnames = [tk.rstrip('/').split('/')[-1] for tk in task_dirs]

        ## NOTE task name to input to the video model
        self.task_list = []
        for tk in task_dirs:
            ## e.g., 'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate'
            ## or e.g., 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet'
            tk = tk.rstrip('/').split('/')[-1]

            ## Find the pattern and replace it with an empty string
            ## '.*?': lazy match anything; 
            pattern = r'.*?SCENE\d+_'
            ## put_the_red_mug_on_the_left_plate
            tk = re.sub(pattern, '', tk, count=1)
            tk = tk.replace("_", " ")
            self.task_list.append(tk)
        
        # pdb.set_trace()

        sequence_dirs = []
        tk_names_input = []


        for i_t, task_dname in enumerate(task_dirnames):
            ## ['datasets_img//libero_90_65To72/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate/000/',...]
            ep_dirs = glob(f"{path}/{dataset_name}/{task_dname}/*/", recursive=False)

            ## process task_dname to 'tk_input', a name that can be input to the video model
            pattern = r'.*?SCENE\d+_'
            tk_input = re.sub(pattern, '', task_dname, count=1)
            tk_input = tk_input.replace("_",  " ")
            print(f'task_dname: {task_dname}')
            print(f'tk_input: {tk_input}')
            # pdb.set_trace()

            tmp_fn_1 = lambda x: int( (x.rstrip('/').split('/')[-1]) )
            ep_dirs = sorted( ep_dirs, key=tmp_fn_1 )
            
            ## select the given episodes
            ep_dirs = ep_dirs[ self.use_episode_range[0]:self.use_episode_range[1] ]
            assert len(ep_dirs) == self.num_ep_per_task
            print(f'ep_dirs 0,-1:', ep_dirs[0][60:], ep_dirs[-1][60:])
            

            sequence_dirs.extend(ep_dirs)
            tk_names_input.extend( [tk_input,] * len(ep_dirs) )

            # pdb.set_trace()

        assert len(sequence_dirs) == len(tk_names_input)
        
        return sequence_dirs, tk_names_input
        


if __name__ == "__main__":
    
    ## Testing
    root_dir = '../datasets/libero/datasets_img' ## might need to change to your own path
    dset_name = 'libero_90_65To72'

    dataset = Sequential_Libero_Dataset_V2_Subgoal(root_dir, dset_name, 
                                use_episode_range=(0,10),num_ep_per_task=10,
                                last_frame_range_len=10,)
    
    x, x_cond, task = dataset[2]
    print(x.shape)
    print(x_cond.shape)
    print(task)

