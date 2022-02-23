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
from dataset import get_2d_dataset, get_random_fixed_2d_dataset
from models.face_gen_former import FaceGenFormer
from utils.model_serializer import ModelSerializer
from utils.save_data import save_images

def compute_visuals(data_dict, output):
    ## we combine input, ref, output, gt together
    input = data_dict['input_image'][:, :, :3, :, :]
    ref = data_dict['input_image'][:, :, 3:, :, :]
    gt = data_dict['gt_face_image']
    output_vis = torch.concat([input, ref, output, gt], dim=-1)
    return output_vis


class Trainer:
    def __init__(self, config) -> None:
        ## Define some parameters
        self.device = torch.device("cuda")
        self.config = config

        ## 1) Define the dataloader
        self.train_dataloader = get_2d_dataset(config['dataset'], split="train")
        print(f"The training dataloader length is {len(self.train_dataloader)}")
        
        ## 2) Define the model and optimizer
        self.model = FaceGenFormer(config, self.device).to(self.device)
        
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=1e-4)

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
        
        # Get fixed batch data for visualization
        vis_training_data = get_random_fixed_2d_dataset(self.config['dataset'], split='train', num_sequences=2)

        ## 3) ========= Start training ======================
        for epoch in range(start_epoch, self.config['epoch_num'] + 1):
            
            prog_bar = tqdm(self.train_dataloader)
            for batch_data in prog_bar:
                train_loss = self._train_step(batch_data)
                
                prog_bar.set_description(f"Epoch: {epoch} | Iter: {global_step} | Training_Loss: {train_loss.item()}")
                
                ## Logging by tensorboard
                self.tb_writer.add_scalar("training_loss", train_loss.item(), global_step)

                global_step += 1

            ## Start Validation TODO


            ## Saving model
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
            
            ## Visualization
            for idx, data in enumerate(vis_training_data):
                data_dict = {}
                for key, value in data.items():
                    data_dict[key] = value[None]

                output = self._test_step(data_dict)
                
                output_vis = compute_visuals(data_dict, output)
                save_images(output_vis, self.config['checkpoint_dir'], epoch, name=f"{idx:03d}")
                
        print("Training Done")    

    def _test_step(self, data_dict):
        self.model.eval()
        
        with torch.no_grad():
            ## Move to GPU
            for key, value in data_dict.items():
                data_dict[key] = value.to(self.device)
            
            ## Build the input
            masked_gt_image = data_dict['gt_face_image'].clone().detach() # (B, T, 3, H, W)
            masked_gt_image[:, :, :, masked_gt_image.shape[3]//2:] = 0.
            data_dict['input_image'] = torch.concat([masked_gt_image, data_dict['ref_face_image']], dim=2) # (B, T, 6, H, W)

            ## Forward the network
            model_output = self.model(data_dict) # (B, T, 3, H, W)

        return model_output    
        

    def _train_step(self, data_dict):
        self.model.train()

        self.optimizer.zero_grad()

        ## Move to GPU
        for key, value in data_dict.items():
            data_dict[key] = value.to(self.device)

        ## Build the input
        masked_gt_image = data_dict['gt_face_image'].clone().detach() # (B, T, 3, H, W)
        masked_gt_image[:, :, :, masked_gt_image.shape[3]//2:] = 0.
        data_dict['input_image'] = torch.concat([masked_gt_image, data_dict['ref_face_image']], dim=2)

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
    

def main():
    #========= Loading Config =========#
    config = OmegaConf.load('./config/config_2d.yaml')
    
    #========= Create Model ============#
    model = Trainer(config)
    model.train()


if __name__ == "__main__":
    main()