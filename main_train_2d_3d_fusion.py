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
from dataset import get_2d_3d_dataset, get_random_fixed_2d_3d_dataset
from models.face_2d_3d_fusion import Face2D3DFusion
from omegaconf import OmegaConf


config = OmegaConf.load('./config/config_2d_3d_fusion.yaml')

model = Face2D3DFusion(config)

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

    predictions = trainer.fit(model, train_dataloader, val_dataloader)
else:
    test_dataloader = get_random_fixed_2d_3d_dataset(config['dataset'], split="val", num_sequences=1)
    print(f"The testing dataloader length is {len(test_dataloader)}")

    model = model.load_from_checkpoint("work_dir/train_2d_3d_fusion/lightning_logs/version_0/checkpoints_test/epoch=31-step=27139.ckpt", 
                                       config=config)
    
    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'], logger=None)
    trainer.test(model, test_dataloader)