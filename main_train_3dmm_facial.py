'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 10:52:25
Email: haimingzhang@link.cuhk.edu.cn
Description: Train the 3DMM generation inspired by FACIAL
'''

import torch
import os.path as osp
import pytorch_lightning as pl
from dataset import get_2d_3d_dataset, get_random_fixed_2d_3d_dataset
from models.face_3dmm_former import Face3DMMFormer
from omegaconf import OmegaConf


config = OmegaConf.load('./config/config_3dmm_facial.yaml')

model = Face3DMMFormer(config)

if not config['test_mode']:
    ## ======================= Training ======================= ##
    ## 1) Define the dataloader
    train_dataloader = get_2d_3d_dataset(config['dataset'], split="train")
    print(f"The training dataloader length is {len(train_dataloader)}")

    val_dataloader = get_2d_3d_dataset(config['dataset'], split='val')
    print(f"The validation dataloader length is {len(val_dataloader)}")

    # trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'])
    trainer = pl.Trainer(gpus=4, default_root_dir=config['checkpoint_dir'], accelerator="gpu", strategy="ddp")

    predictions = trainer.fit(model, train_dataloader)
else:
    test_dataloader = get_random_fixed_2d_3d_dataset(config['dataset'], split="val", num_sequences=2)
    print(f"The testing dataloader length is {len(test_dataloader)}")

    model = model.load_from_checkpoint(osp.join(config['checkpoint_dir'], "lightning_logs/version_3/checkpoints/epoch=232-step=6290.ckpt"), 
                                       config=config)
    
    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'])
    trainer.test(model, test_dataloader)
