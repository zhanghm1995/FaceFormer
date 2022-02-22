'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-19 09:42:33
Email: haimingzhang@link.cuhk.edu.cn
Description: Serialize the network model, e.g. save and load the model for training and inference
'''

import os.path as osp
from pickle import NONE
import torch
import shutil

class ModelSerializer(object):
    def __init__(self, latest_ckpt_fpath, best_ckpt_fpath=None) -> None:
        self.latest_ckpt_fpath = latest_ckpt_fpath
        self.best_ckpt_fpath = best_ckpt_fpath

    def save(self, state, is_best):
        self.save_ckp(
            state, is_best, 
            checkpoint_path=self.latest_ckpt_fpath, 
            best_model_path=self.best_ckpt_fpath)
    
    def restore(self, model, optimizer, load_latest=True, load_best=False):
        start_epoch, valid_loss_min = 1, 1000.0

        if load_best and osp.exists(self.best_ckpt_fpath):
            start_epoch, global_step, valid_loss_min = self.load_ckp(
                self.best_ckpt_fpath, model, optimizer)
            print(f"[INFO] Load best checkpoint start_epoch: {start_epoch}")
        elif load_latest and osp.exists(self.latest_ckpt_fpath):
            start_epoch, global_step, valid_loss_min = self.model_serializer.load_ckp(
                self.latest_ckpt_fpath, model, optimizer)
            print(f"[INFO] Load latest checkpoint start_epoch: {start_epoch}")
        else:
            start_epoch, global_step = 1, 1
            print("[WARNING] Train from scratch!")
        return start_epoch, global_step, valid_loss_min

    def save_ckp(slef, state, is_best, checkpoint_path, best_model_path):
        """
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model
        """
        f_path = checkpoint_path
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, f_path)
        # if it is a best model, min validation loss
        if is_best:
            best_fpath = best_model_path
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(f_path, best_fpath)

    def load_ckp(self, checkpoint_fpath, model, optimizer):
        """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into       
        optimizer: optimizer we defined in previous training
        """
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']
        # return model, optimizer, epoch value, min validation loss 
        return checkpoint['epoch'], checkpoint['global_step'], valid_loss_min