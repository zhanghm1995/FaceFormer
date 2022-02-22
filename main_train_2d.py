'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-22 11:09:34
Email: haimingzhang@link.cuhk.edu.cn
Description: Train the 2D image generation transformer
'''

import os
import os.path as osp
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import get_2d_dataset
from models.face_gen_former import FaceGenFormer
from utils.model_serializer import ModelSerializer


def test_dataloader(config):
    #========= Loading Dataset =========#
    train_dataloader = get_2d_dataset(config['dataset'], split="train")
    print(len(train_dataloader))

    for data_batch in train_dataloader:
        print(data_batch.shape)
        break

class Trainer:
    def __init__(self, config) -> None:
        ## Define some parameters
        self.device = torch.device("cuda")
        self.config = config

        ## 1) Define the dataloader
        self.train_dataloader = get_2d_dataset(config['dataset'], split="train")
        print(f"The training dataloader length is {len(self.train_dataloader)}")
        
        ## 2) Define the model and optimizer
        self.model = FaceGenFormer(config, self.device)
        
        self.optimizer = optim.Adam([p for p in self.model.parameters()], lr=1e-4)

        ## 3) Define the loss
        self.criterion = nn.L1Loss().to(self.device)
        
        ## 4) Logging
        self.model_serializer = ModelSerializer(
            osp.join(self.config['checkpoint_dir'], "latest_model.ckpt"),
            osp.join(self.config['checkpoint_dir'], "best_model.ckpt"))

    def train(self):
        ## 1) Define the logging
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(osp.join(self.config['checkpoint_dir'], "logdir"))

        ## 2) Restore the network
        start_epoch, global_step = 1, 1
        start_epoch, global_step, _ = \
            self.model_serializer.restore(self.model, self.optimizer, load_latest=True)
        
        ## 3) ========= Start training ======================
        for epoch in range(start_epoch, self.config['epoch_num'] + 1):
            
            prog_bar = tqdm(self.train_dataloader)
            for batch_data in prog_bar:
                train_loss = self._train_step(batch_data)
                
                prog_bar.set_description(f"Epoch: {epoch} | Iter: {global_step} | Training_Loss: {train_loss.item()}")

                global_step += 1

            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'global_step': global_step + 1,
                    'valid_loss_min': 1000.0,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }   
                self.model_serializer.save(checkpoint, is_best=False)
                print(f"Saving checkpoint in epoch {epoch}")
        
        print("Training Done")    


    def _train_step(self, data_dict):
        self.model.train()

        self.optimizer.zero_grad()

        ## Forward the network
        model_output = self.model(data_dict)

        ## Calculate the loss
        loss = self.criterion(model_output, data_dict['gt_face_image'])

        ## Loss backward and update network parameters
        loss.backward()
        self.optimizer.step()
        
        return loss

    def _val_step(self):
        pass
    
    def _test_step(self):
        pass

def main():
    #========= Loading Config =========#
    config = OmegaConf.load('./config/config_2d.yaml')
    test_dataloader(config)


if __name__ == "__main__":
    main()