from pathlib import Path
from models import utils as mutils
from models.Unet import Unet 
import time
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
from utils import min_max_normalize,restore_checkpoint

def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    # fname = '001'
    fname1 = args.image1
    filename = f'./samples/{fname1}.npy'
    
    fname2 = args.image2
    filename2 = f'./samples/{fname2}.npy'
    
    
    print('initaializing...')
    configs = importlib.import_module(f"configs.Unet.unet_ds")
    config = configs.get_config()
    batch_size = 1

    # Read data
    img1 = torch.from_numpy(min_max_normalize(np.load(filename)).astype(np.float32))
    img1  = img1.view(1, 1, config.data.image_size1, config.data.image_size2)
    img2 = torch.from_numpy(min_max_normalize(np.load(filename2)).astype(np.float32))
    img2 = img2.view(1, 1, config.data.image_size1, config.data.image_size2)
    
    img1,img2 = img1.to(config.device),img2.to(config.device)
    

    ckpt_filename = f"./weights/checkpoint_90.pth"

    # create model and load checkpoint
    model = mutils.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True,skip_optimizer=True)
    ema.copy_to(model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input1','input2','recon']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)
    ###############################################
    # 2. Inference
    ###############################################
    print(f'Beginning inference')
    tic = time.time()
    x = model(img1,img2)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################
    plt.imsave(str(save_root / 'input1' / fname1) + '.png', img1.squeeze().cpu().detach().numpy(), cmap='gray')
    plt.imsave(str(save_root / 'input2' / fname2) + '.png', img2.squeeze().cpu().detach().numpy(), cmap='gray')
    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname1) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname1) + '.png', recon, cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', type=str, help='which first 25um to use for ds', required=True)
    parser.add_argument('--image2', type=str, help='which second 25um to use for ds', required=True)
    parser.add_argument('--save_dir', default='./results')
    return parser

if __name__ == "__main__":
    main()