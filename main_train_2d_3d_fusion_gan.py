'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-28 23:57:46
Email: haimingzhang@link.cuhk.edu.cn
Description: The main training entrance 
'''

import argparse
import pytorch_lightning as pl
from dataset import get_2d_3d_dataset, get_random_fixed_2d_3d_dataset
from models.face_2d_3d_fusion_gan import Face2D3DFusionGAN
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import get_git_commit_id
from models import get_model


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/config_2d_3d_fusion_new.yaml', help='the config file path')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')
    parser.add_argument('--checkpoint_dir', type=str, nargs='?', const="debug")
    parser.add_argument('--checkpoint', type=str, default=None, help="the pretrained checkpoint path")
    parser.add_argument('--test_mode', action='store_true', help="whether is a test mode")

    args = parser.parse_args()
    config = OmegaConf.load(args.cfg)

    if args.checkpoint_dir is None: # use the yaml value if don't specify the checkpoint_dir argument
        args.checkpoint_dir = config.checkpoint_dir
    
    config.update(vars(args)) # override the configuration using the value in args

    print(OmegaConf.to_yaml(config, resolve=True))
    
    try:
        config['commit_id'] = get_git_commit_id()
    except:
        print("[WARNING] Couldn't get the git commit id")
    return config


config = parse_config()

## Create model
model = get_model(config['model_name'], config)

## Load pre-trained check-point
if config.checkpoint is None:
    print(f"[WARNING] Train from scratch!")
else:
    print(f"Load pretrained model from {config.checkpoint}")
    model = model.load_from_checkpoint(config.checkpoint, config=config)


if not config['test_mode']:
    print(f"{'='*25} Start Traning, Good Luck! {'='*25}")

    ## ======================= Training ======================= ##
    ## 1) Define the dataloader
    train_dataloader = get_2d_3d_dataset(config['dataset'], split="train")
    print(f"The training dataloader length is {len(train_dataloader)}")

    val_dataloader = get_2d_3d_dataset(config['dataset'], split='val')
    print(f"The validation dataloader length is {len(val_dataloader)}")

    # checkpoint_callback = ModelCheckpoint(monitor='precision/test', mode='max', save_last=True,
    #                                       save_top_k=cfg.save_top_k)

    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'],
                         max_epochs=config.max_epochs,
                         check_val_every_n_epoch=config.check_val_every_n_epoch)
    # trainer = pl.Trainer(gpus=4, default_root_dir=config['checkpoint_dir'], accelerator="gpu", strategy="ddp")

    predictions = trainer.fit(model, train_dataloader, val_dataloader, 
                              ckpt_path=config.checkpoint)
else:
    test_dataloader = get_random_fixed_2d_3d_dataset(config['dataset'], split="val", num_sequences=1)
    print(f"The testing dataloader length is {len(test_dataloader)}")

    model = model.load_from_checkpoint("work_dir/train_2d_3d_fusion/lightning_logs/version_0/checkpoints_test/epoch=121-step=103160.ckpt", 
                                       config=config)
    
    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'], logger=None)
    trainer.test(model, test_dataloader)