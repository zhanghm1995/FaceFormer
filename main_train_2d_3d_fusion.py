'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-28 23:57:46
Email: haimingzhang@link.cuhk.edu.cn
Description: The main training entrance 
'''

import torch
import os.path as osp
import pytorch_lightning as pl
from dataset import get_2d_3d_dataset, get_random_fixed_2d_3d_dataset, get_test_2d_3d_dataset
from models.face_2d_3d_fusion import Face2D3DFusion
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import get_git_commit_id

config = OmegaConf.load('./config/config_2d_3d_fusion_mmt_with_ref.yaml')

config['commit_id'] = get_git_commit_id()

## Create model
if config.checkpoint is None:
    print(f"[WARNING] Train from scratch!")
    model = Face2D3DFusion(config)
else:
    print(f"Load pretrained model from {config.checkpoint}")
    model = Face2D3DFusion(config).load_from_checkpoint(config.checkpoint, config=config)

if not config['test_mode']:
    print(f"{'='*25} Start Traning, Good Luck! {'='*25}")

    ## ======================= Training ======================= ##
    ## 1) Define the dataloader
    train_dataloader = get_2d_3d_dataset(config['dataset'], split="train")
    print(f"The training dataloader length is {len(train_dataloader)}")

    val_dataloader = get_2d_3d_dataset(config['dataset'], split='val')
    print(f"The validation dataloader length is {len(val_dataloader)}")

    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'],
                         max_epochs=config.max_epochs,
                         check_val_every_n_epoch=config.check_val_every_n_epoch)
    # trainer = pl.Trainer(gpus=4, default_root_dir=config['checkpoint_dir'], accelerator="gpu", strategy="ddp")

    ## Resume the training state
    predictions = trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=config.checkpoint)
else:
    config['dataset']['audio_path'] = "data/audio_samples/slogan_english_16k.wav"
    config['dataset']['video_path'] = "data/id00002/obama_weekly_029/face_image"
    config['dataset']['face_3d_params_path'] = "data/id00002/obama_weekly_029/deep3dface.npz"

    test_dataloader = get_test_2d_3d_dataset(config['dataset'])
    print(f"The testing dataloader length is {len(test_dataloader)}")

    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'], logger=None)
    trainer.test(model, test_dataloader)