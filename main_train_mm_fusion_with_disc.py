'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-22 11:09:34
Email: haimingzhang@link.cuhk.edu.cn
Description: Train the Multi-Modal fusion transformer with Discriminators
'''

import os
import os.path as osp
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from dataset import get_2d_3d_dataset, get_random_fixed_2d_3d_dataset
from models.mm_fusion_transformer import MMFusionFormer
from models.face_gen_module import FaceGenModule
from utils.model_serializer import ModelSerializer
from typing import Dict
from utils.save_data import save_images

def compute_visuals(data_dict, output):
    ## we combine input, ref, output, gt together
    input = data_dict['input_image'][:, :, :3, :, :]
    ref = data_dict['input_image'][:, :, 3:, :, :]
    gt = data_dict['gt_face_image']
    output_vis = torch.concat([input, ref, output, gt], dim=-1)
    return output_vis


def compute_losses(data_input: dict, model_output: dict, criterion: dict, config=None) -> dict:
    """Compute all losses

    Args:
        data_input (dict): input data dictionary
        model_output (dict): model network output
        criterion (dict): all criterions used to compute losses
        config (_type_, optional): config parameters. Defaults to None.

    Returns:
        dict: computed losses dictionary
    """
    total_loss = 0.0

    face_2d_image_criterion = criterion['face_2d_image']
    face_2d_image_loss = face_2d_image_criterion(model_output['face_image'], data_input['gt_face_image'])
    total_loss += face_2d_image_loss

    face_3d_params_criterion = criterion['face_3d_params']
    face_3d_params_loss = face_3d_params_criterion(model_output['face_3d_params'], data_input['face_3d_params'])

    total_loss += face_3d_params_loss

    return {'total_loss': total_loss,
            'face_2d_image_loss': face_2d_image_loss,
            'face_3d_params_loss': face_3d_params_loss}


def get_loss_description_str(loss_dict):
    assert isinstance(loss_dict, dict)

    description_str = ""
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            description_str += f"{key}: {value.item():0.4f} "
        else:
            description_str += f"{key}: {value:0.4f} "
    return description_str


def add_tensorboard_scalar(writer: SummaryWriter, loss_dict: Dict, split, step):
    assert isinstance(loss_dict, dict)

    for key, val in loss_dict.items():
        writer.add_scalar(f"{split}/{key}", val, step)


class Trainer:
    def __init__(self, config) -> None:
        ## Define some parameters
        self.device = torch.device("cuda")
        self.config = config

        ## 1) Define the dataloader
        self.train_dataloader = get_2d_3d_dataset(config['dataset'], split="train")
        print(f"The training dataloader length is {len(self.train_dataloader)}")
        self.val_dataloader = get_2d_3d_dataset(config['dataset'], split='val')
        print(f"The validation dataloader length is {len(self.val_dataloader)}")
        
        ## 2) Define the model
        self.model = FaceGenModule(config, self.device)
        
        ## 3) Logging
        self.model_G_serializer = ModelSerializer(
            osp.join(self.config['checkpoint_dir'], "latest_G_model.ckpt"),
            osp.join(self.config['checkpoint_dir'], "best_G_model.ckpt"))
        self.model_D_serialzier = ModelSerializer(
            osp.join(self.config['checkpoint_dir'], "latest_D_model.ckpt"),
            osp.join(self.config['checkpoint_dir'], "best_D_model.ckpt"))

    def train(self):
        ## 1) Define the logging
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(osp.join(self.config['checkpoint_dir'], "logdir"))

        ## Save the config parameters
        OmegaConf.save(self.config, osp.join(self.config['checkpoint_dir'], "config_train.yaml"))

        ## 2) Restore the network
        start_epoch, global_step = 1, 1
        start_epoch, global_step, _ = \
            self.model_G_serializer.restore(self.model.net_G, self.model.optimizer_G, load_latest=True)
        self.model_D_serialzier.restore(self.model.net_D, self.model.optimizer_D, load_latest=True)
        
        # Get fixed batch data for visualization
        vis_val_data = get_random_fixed_2d_3d_dataset(self.config['dataset'], split='val', num_sequences=2)

        min_valid_loss, avg_val_loss = 1000.0, 2000.0

        ## 3) ========= Start training ======================
        for epoch in range(start_epoch, self.config['epoch_num'] + 1):
            prog_bar = tqdm(self.train_dataloader)
            for batch_data in prog_bar:
                train_loss = self.model.train(batch_data)
                loss_description_str = get_loss_description_str(train_loss)

                description_str = (f"Training: Epoch: {epoch} | Iter: {global_step} | "
                                   + loss_description_str)
                
                prog_bar.set_description(description_str)
                
                ## Logging by tensorboard
                add_tensorboard_scalar(self.tb_writer, train_loss, "train", global_step)

                global_step += 1

            ## Start Validation
            if epoch % 4 == 0:
                print("================= Start validation ==================")
                avg_val_loss = self._val_step(epoch, global_step)
                 ## Logging by tensorboard
                add_tensorboard_scalar(self.tb_writer, avg_val_loss, "val", global_step)

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
                    
                    output_vis = compute_visuals(data_dict, output['face_image'])
                    save_images(output_vis, osp.join(self.config['checkpoint_dir'], "vis"), epoch, name=f"{idx:03d}")
                    
        print("Training Done")    

    def test(self):
        print("================ Start testing ======================")
        ## 1) Restore the network
        start_epoch, global_step = 1, 1
        start_epoch, global_step, _ = \
            self.model_serializer.restore(self.model, self.optimizer, load_latest=True)
        
        # Get fixed batch data for visualization
        vis_val_data = get_random_fixed_2d_3d_dataset(self.config['dataset'], split='val', num_sequences=2)

        ## 2) ========= Start training ======================
        epoch = start_epoch
        ## Visualization
        for idx, data in enumerate(vis_val_data):
            data_dict = {}
            for key, value in data.items():
                data_dict[key] = value[None]

            output = self._test_step(data_dict)
            
            output_vis = compute_visuals(data_dict, output['face_image'])
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
        all_losses = compute_losses(data_dict, model_output, self.criterion)

        total_loss = all_losses['total_loss']

        ## Loss backward and update network parameters
        total_loss.backward()
        self.optimizer.step()
        
        return all_losses

    def _val_step(self, epoch, global_step, autoregressive=False):

        def calc_avg_loss(loss_dict_list):
            assert len(loss_dict_list) != False, "input list length is 0"

            avg_loss_dict = dict()

            all_loss_keys = loss_dict_list[0].keys()

            for key in all_loss_keys:
                loss_list = [loss_dict[key] for loss_dict in loss_dict_list]
                avg_loss_dict[key] = sum(loss_list) / len(loss_list)

            return avg_loss_dict

        self.model.eval()
        
        with torch.no_grad():
            val_loss_list = []

            prog_bar = tqdm(self.val_dataloader)
            for batch_data in prog_bar:
                val_loss_dict = self.model.validate(batch_data)

                loss_description_str = get_loss_description_str(val_loss_dict)

                description_str = (f"Validation: Epoch: {epoch} | Iter: {global_step} | "
                                   + loss_description_str)

                prog_bar.set_description(description_str)

                val_loss_list.append(val_loss_dict)

            ## Get the average validation loss
            average_val_loss = calc_avg_loss(val_loss_list)
            loss_description_str = get_loss_description_str(average_val_loss)
            print(f"Epoch: {epoch} | Iter: {global_step} | Average Validation Loss: {average_val_loss}")
        return average_val_loss
    

def main():
    #========= Loading Config =========#
    config = OmegaConf.load('./config/config_2d_3d_with_disc.yaml')
    
    #========= Create Model ============#
    model = Trainer(config)
    model.train()


if __name__ == "__main__":
    main()