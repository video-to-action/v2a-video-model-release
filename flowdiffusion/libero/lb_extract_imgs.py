import sys; sys.path.append('./')
from libero.libero import benchmark, get_libero_path, set_libero_default_path
import os
from termcolor import colored
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['MUJOCO_GL'] = 'egl'
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
import h5py, pdb
import mediapy as mpy
import os.path as osp
from diffuser.utils import save_imgs_1, save_json
"""
    This file will extract the images in the downloaded official Libero hdf5 files 
    to a local folder for training the video models.
    It might take 10-20 minutes. Please also check the note in the comment below.
    Extracted Data for the 8 tasks are around 920M.
"""

def main():
    
    benchmark_dict = benchmark.get_benchmark_dict()
    print(benchmark_dict)

    benchmark_instance = benchmark_dict["libero_90"]() # Not exist libero_100

    num_tasks = benchmark_instance.get_num_tasks()
    print(f"Number of tasks in the benchmark {benchmark_instance.name}: {num_tasks}")

    
    ## -------------------------------------------------------------------
    """ NOTE:
    all_demo_files is a list of paths for data of all 90 tasks in libero_90,
    if you only download the required tasks, please modify the path accordingly.
    Please do not change the name of those hdf5 files.
    """

    datasets_default_path = get_libero_path("datasets")
    all_demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]

    ## ids of the task of interest. You can see a complete list in the Libero Codebase
    ## tasks in Living Room Scene 5 and 6 as in the paper
    idxs = [65,66,67,68,69,70,71,72]
    ## demo to be extracted to images from hdf5
    demo_files = [all_demo_files[idx] for idx in idxs]

    ## -------------------------------------------------------------------

    # pdb.set_trace()
    
    ## NOTE: save directory of the extracted images, relative to the root dir of the repo
    cur_work_dir = os.getcwd()
    assert 'flowdiffusion' == cur_work_dir[-13:], 'make sure the relative parent_dir is correct'
    parent_dir = '../datasets/libero/datasets_img/'
    dataset_name = 'libero_90_65To72' ## used for subfolder creation


    # pdb.set_trace()


    def extract_demos_of_1_task(demo_file, parent_dir, dataset_name):
        '''
        save the 50 demos corresponding to one task (one of libero-90)
        demo_file: the hdf5 file of the task
        '''
        d_file = demo_file
        # dir_last_two = '/'.join( d_file.split('/')[-2:])
        # dataset_name = d_file.split('/')[-2] ## use from args instead
        # pdb.set_trace()

        task_name = d_file.split('/')[-1]
        task_name = task_name[:task_name.find('_demo.hdf5')]
        ## libero_90 LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate
        # print( dataset_name, task_name )
        print( osp.split(d_file) )
        tk_save_path = osp.join( parent_dir, dataset_name, task_name )
        print( 'tk_save_path:', tk_save_path )

        with h5py.File(d_file, "r") as f:
            tmp_fn_1 = lambda x: int(x.split('_')[1])
            ## keys: ['demo_0', 'demo_1', 'demo_2', ...]
            keys = sorted(f['data'].keys(), key=tmp_fn_1)
            demo_idxs = [tmp_fn_1(key) for key in keys]
            print(keys[:20])
            print(demo_idxs[:20])
            print(f'number of demos:', len(keys))
            # print(f['data/demo_0'].keys())
            # print(f['data/demo_0/actions'][()].shape)
            for i_k, key in enumerate(keys):
                imgs_gt = f[f"data/{key}/obs/agentview_rgb"][()][:, ::-1, :, :]
                ee_states = f[f"data/{key}/obs/ee_states"][()]
                acts_gt = f[ f'data/{key}/actions' ][()]
                print(imgs_gt.shape, imgs_gt.dtype)
                print(acts_gt.shape, acts_gt.dtype)
                ## save path of this rollout
                ## 'datasets_img/libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate/000/'
                demo_save_path = osp.join(tk_save_path, f'{demo_idxs[i_k]:03d}')
                os.makedirs(demo_save_path, exist_ok=True)
                # print(demo_save_path)

                ## save imgs
                imgs_path = []
                for i_im in range(len(imgs_gt)):
                    imgs_path.append(  osp.join(demo_save_path, 'agentview_rgb', f'{i_im:03d}.png') )
                save_imgs_1(imgs_gt, imgs_path)

                ## save acts
                acts_path = osp.join(demo_save_path, 'action.json')
                save_json(acts_gt.tolist(), acts_path)

                ## save ee_states
                ee_states_path = osp.join(demo_save_path, 'ee_states.json')
                save_json(ee_states.tolist(), ee_states_path)

                print(imgs_path[0])
            
            ## -------------------------------------------
            ## ----- Finished save all e.g. 50 demos -----


    for d_file in demo_files:
        extract_demos_of_1_task(d_file, parent_dir, dataset_name)




if __name__ == '__main__':
    main()