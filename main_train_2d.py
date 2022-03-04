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
        self.val_dataloader = get_2d_dataset(config['dataset'], split='val')
        print(f"The validation dataloader length is {len(self.val_dataloader)}")
        
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

        ## Save the config parameters
        OmegaConf.save(self.config, osp.join(self.config['checkpoint_dir'], "config_train.yaml"))

        ## 2) Restore the network
        start_epoch, global_step = 1, 1
        start_epoch, global_step, _ = \
            self.model_serializer.restore(self.model, self.optimizer, load_latest=True)
        
        # Get fixed batch data for visualization
        vis_training_data = get_random_fixed_2d_dataset(self.config['dataset'], split='train', num_sequences=2)

        vis_val_data = get_random_fixed_2d_dataset(self.config['dataset'], split='val', num_sequences=2)

        min_valid_loss, avg_val_loss = 1000.0, 2000.0

        ## 3) ========= Start training ======================
        for epoch in range(start_epoch, self.config['epoch_num'] + 1):
            
            prog_bar = tqdm(self.train_dataloader)
            for batch_data in prog_bar:
                train_loss = self._train_step(batch_data)
                
                prog_bar.set_description(f"Epoch: {epoch} | Iter: {global_step} | Training_Loss: {train_loss.item()}")
                
                ## Logging by tensorboard
                self.tb_writer.add_scalar("training_loss", train_loss.item(), global_step)

                global_step += 1

            ## Start Validation
            if epoch % 4 == 0:
                print("================= Start validation ==================")
                avg_val_loss = self._val_step(epoch, global_step)
                 ## Logging by tensorboard
                self.tb_writer.add_scalar("val_loss", avg_val_loss, global_step)

            ## Saving model
            if epoch % 4 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'global_step': global_step + 1,
                    'valid_loss_min': 1000.0,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }   

                if avg_val_loss < min_valid_loss:
                    self.model_serializer.save(checkpoint, is_best=True)
                    print(f"Saving best checkpoint in epoch {epoch} with best validation loss: {avg_val_loss}")
                    min_valid_loss = avg_val_loss
                else:
                    self.model_serializer.save(checkpoint, is_best=False)
                    print(f"Saving latest checkpoint in epoch {epoch}")
                    
            
            ## Visualization
            if epoch % 4 == 0:
                for idx, data in enumerate(vis_val_data):
                    data_dict = {}
                    for key, value in data.items():
                        data_dict[key] = value[None]

                    output = self._test_step(data_dict)
                    
                    output_vis = compute_visuals(data_dict, output)
                    save_images(output_vis, osp.join(self.config['checkpoint_dir'], "vis"), epoch, name=f"{idx:03d}")
                    
        print("Training Done")    

    def test(self):
        print("================ Start testing ======================")
        ## 1) Restore the network
        start_epoch, global_step = 1, 1
        start_epoch, global_step, _ = \
            self.model_serializer.restore(self.model, self.optimizer, load_latest=True)
        
        # Get fixed batch data for visualization
        vis_val_data = get_random_fixed_2d_dataset(self.config['dataset'], split='val', num_sequences=2)

        ## 2) ========= Start training ======================
        epoch = start_epoch
        ## Visualization
        for idx, data in enumerate(vis_val_data):
            data_dict = {}
            for key, value in data.items():
                data_dict[key] = value[None]

            output = self._test_step(data_dict)
            
            output_vis = compute_visuals(data_dict, output)
            save_images(output_vis, osp.join(self.config['checkpoint_dir'], "vis"), epoch, name=f"{idx:03d}")
                    
        print("Training Done")

    def _test_step(self, data_dict, autoregressive=False):
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
            if autoregressive:
                model_output = self.model.inference(data_dict)
            else:
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

    def _val_step(self, epoch, global_step, autoregressive=False):
        self.model.eval()
        
        with torch.no_grad():
            val_loss_list = []

            prog_bar = tqdm(self.val_dataloader)
            for batch_data in prog_bar:
                ## Move to GPU
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                
                ## Build the input
                masked_gt_image = batch_data['gt_face_image'].clone().detach() # (B, T, 3, H, W)
                masked_gt_image[:, :, :, masked_gt_image.shape[3]//2:] = 0.
                batch_data['input_image'] = torch.concat([masked_gt_image, batch_data['ref_face_image']], dim=2) # (B, T, 6, H, W)

                ## Forward the network
                if autoregressive:
                    model_output = self.model.inference(batch_data)
                else:
                    model_output = self.model(batch_data) # (B, T, 3, H, W)

                val_loss = self.criterion(model_output, batch_data['gt_face_image'])

                prog_bar.set_description(f"Epoch: {epoch} | Iter: {global_step} | Validation Loss: {val_loss.item()}")

                val_loss_list.append(val_loss.item())

            average_val_loss = sum(val_loss_list) / len(val_loss_list)
            print(f"Epoch: {epoch} | Iter: {global_step} | Average Validation Loss: {average_val_loss}")
        return average_val_loss
    

def main():
    #========= Loading Config =========#
    config = OmegaConf.load('./config/config_2d.yaml')
    
    #========= Create Model ============#
    model = Trainer(config)
    model.train()


if __name__ == "__main__":
    main()