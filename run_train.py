'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-13 15:26:22
Email: haimingzhang@link.cuhk.edu.cn
Description: script to run training pipeline
'''
import os
import os.path as osp
import numpy as np
import logging
import torch
import torch.nn as nn
from torch import optim
from config_parser import get_configs
from dataset.voca_dataset import DataHandler, Batcher
from test_voca_dataset import one_hot
from wav2vec2 import FaceFormer, FaceFormerV2


class Trainer(object):
    def __init__(self, config, batcher: Batcher) -> None:
        self.device = torch.device("cuda")

        self.batcher = batcher
        self.config = config
        
        self.model = FaceFormerV2(self.config, self.device).to(self.device)

        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                     lr=1e-4)
        
        self.criterion = nn.MSELoss()

        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(osp.join(self.config['checkpoint_dir'], "logdir"))
    
    def train(self):
        num_train_batches = self.batcher.get_num_batches(self.config['batch_size'])
        global_step = 0

        for epoch in range(1, self.config['epoch_num'] + 1):
            for iter in range(num_train_batches):
                loss = self._training_step()
                
                print(f"Epoch: {epoch} | Iter: {iter} | Global Step: {global_step} | Loss: {loss}")
                self.tb_writer.add_scalar("traing_loss", loss, global_step)
                # if iter % 100 == 0:
                #     val_loss = self._validation_step()
                #     logging.warning("Validation loss: %.6f" % val_loss)
                #     self.tb_writer.add_scalar("validation_loss", val_loss, global_step)
                
                global_step += 1

            # if epoch % 10 == 0:
            #     self._save(global_step)

            # if epoch % 25 == 0:
            #     self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 'training_epoch_%d_iter_%d' % (epoch, iter))
            #                            , data_specifier='training')
            #     self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 'validation_epoch_%d_iter_%d' % (epoch, iter))
            #                            , data_specifier='validation')

    def _prepare_data(self, batch_data_dict, device):
        _, seq_len = batch_data_dict['face_vertices'].shape[:2]

        #======= Prepare the GT face motion ==========#
        batch_data_dict['target_face_motion'] = \
            batch_data_dict['face_vertices'] - np.expand_dims(batch_data_dict['face_template'], axis=1)

        #======== Prepare the subject idx ===========#
        subject_idx = np.expand_dims(np.stack(batch_data_dict['subject_idx']), -1)
        batch_data_dict['subject_idx'] = one_hot(torch.from_numpy(subject_idx.repeat(seq_len, axis=-1))).to(torch.float32)

        for key, value in batch_data_dict.items():
            if key != "subject_idx":
                batch_data_dict[key] = torch.from_numpy(value).type(torch.FloatTensor).to(device)
            else:
                batch_data_dict[key] = value.to(device)
            
    def _training_step(self):
        self.model.train()
        
        ## prepare datas
        batch_data_dict = self.batcher.get_training_batch(self.config['batch_size'])
        self._prepare_data(batch_data_dict, self.device)

        ## forward
        self.optimizer.zero_grad()
        pred_facial_motion = self.model(batch_data_dict)

        pred_facial_vertices = batch_data_dict['face_template'].unsqueeze(1) + pred_facial_motion

        loss = self.criterion(pred_facial_vertices, batch_data_dict['face_vertices'])
        loss.backward()

        return loss
    
    def _validation_step(self):
        self.model.eval()

    def _save(self):
        pass

    def save_model(self, checkpoint, model_name, save_mode="best"):
        if save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')


def main():
    from omegaconf import OmegaConf

    config = OmegaConf.load('./config/config.yaml')

    #========= Loading Dataset =========#
    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)

    #========= Create Model ============#
    model = Trainer(config, batcher)
    model.train()


if __name__ == "__main__":
    main()
