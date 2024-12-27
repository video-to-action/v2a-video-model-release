import sys; sys.path.append('./')
from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import Unet_Libero
from transformers import CLIPTextModel, CLIPTokenizer
import os; import os.path as osp
from libero.libero_datasets_v2 import Sequential_Libero_Dataset_V2
from libero.libero_datasets_v2_sg import Sequential_Libero_Dataset_V2_Subgoal
from torch.utils.data import Subset
import torch, pdb
import diffuser.utils as utils

torch.backends.cudnn.benchmark = True

class Parser(utils.Parser):
    config: str ## Path to the config file
    mode: str = 'train'
    checkpoint_num: int = None # to resume training
    inference_path: str = None
    text: str = None


def main(args):
    valid_n = args.valid_n # 
    sample_per_seq = getattr(args, 'sample_per_seq', 8) ## so the horizon is 8-1=7
    target_size = args.target_size # (128,128)
    train_results_folder = args.savepath
    
    # pdb.set_trace()

    if args.mode == 'inference':
        train_set = valid_set = [None] # dummy
    else:
        ## we use the default avdc sample_model
        sample_mode = getattr(args, 'sample_mode', 'avdc')
        if sample_mode == 'subgoal':
            ds_cls = Sequential_Libero_Dataset_V2_Subgoal
        elif sample_mode == 'avdc':
            ds_cls = Sequential_Libero_Dataset_V2
        else:
            raise NotImplementedError()
        
        utils.print_color(f'{ds_cls=}', c='y')

        ## initialize the train dataset
        train_set = ds_cls(
            path=args.root_dir, 
            dataset_name=args.dset_name,
            sample_per_seq=sample_per_seq, 
            target_size=target_size,
            frameskip=None, randomcrop=args.randomcrop,
            use_episode_range=args.use_episode_range, #(0,30),
            num_ep_per_task=args.num_ep_per_task, # 30,
            last_frame_range_len=args.last_frame_range_len,
            dataset_config=args.dataset_config,
        )

        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)
    
    # for i in range(10):
        # train_set[i]
    # pdb.set_trace()

    unet = Unet_Libero()

    ## check text encoding here, given B sentences, output (B,512)
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    # pdb.set_trace()

    ## Initialize the Video Diffusion Model
    diffusion = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=args.train_num_steps, # 180000,
        save_and_sample_every=4000,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size=args.train_batch_size, # 16
        valid_batch_size =12, # 32
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder =train_results_folder,
        fp16 =True,
        amp=True,
    )

    ## For Resume Training from a specific checkpint
    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    
    if args.mode == 'train':
        trainer.train()
    else:
        ## ------- Below is just for inference sampling --------
        from PIL import Image
        from torchvision import transforms
        import imageio
        import torch
        from os.path import splitext
        text = args.text
        image = Image.open(args.inference_path)
        batch_size = 1
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        ## Seeding
        utils.set_seed(1)
        output = trainer.sample(image.unsqueeze(0), [text], batch_size).cpu()
        output = output[0].reshape(-1, 3, *target_size)
        output = torch.cat([image.unsqueeze(0), output], dim=0)
        root, ext = splitext(args.inference_path)
        # output_gif = root + '_out.gif'
        text_dashed = text.replace(' ', '-')
        output_gif = f'{osp.abspath(train_results_folder)}/plan/{text_dashed}_libero_out.gif'
        os.makedirs(osp.dirname(output_gif) ,exist_ok=True)
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        ## 300ms per frame
        imageio.mimsave(output_gif, output, duration=300, loop=1000)

        print(f'Generated {output_gif}')

        ## ----------------------------------------------------

if __name__ == "__main__":
    
    ## **set dataset in parse_args automatically
    args = Parser().parse_args('diffusion')
    args.sample_steps = 10

    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
    
    assert args.sample_steps <= 100
    main(args)