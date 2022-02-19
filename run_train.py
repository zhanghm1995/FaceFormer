'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-13 15:26:22
Email: haimingzhang@link.cuhk.edu.cn
Description: script to run training pipeline
'''
import os
import os.path as osp
import cv2
import tempfile
import threading
import numpy as np
import logging
from scipy.io import wavfile
import torch
import torch.nn as nn
from torch import optim
from subprocess import call
from config_parser import get_configs
from dataset import get_dataset
from test_voca_dataset import one_hot
from wav2vec2 import FaceFormer, FaceFormerV2
from utils.rendering import render_mesh_helper
from psbody.mesh import Mesh
from utils.model_serializer import ModelSerializer

MIN_MOTION = np.array([-0.00916614, -0.02674509, -0.0166305])
MAX_MOTION = np.array([0.01042878, 0.01583716, 0.01325295])
DIFF_MOTION = MAX_MOTION - MIN_MOTION

def normalize_motion(input):
    output = input - MIN_MOTION[None, None, None, :]
    output = output / DIFF_MOTION[None, None, None, :]
    return output


def denormalize_motion(input):
    output = np.multiply(input, DIFF_MOTION[None, None, None, :])
    output = output + MIN_MOTION[None, None, None, :]
    return output


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))

class Trainer(object):
    def __init__(self, config, batcher) -> None:
        self.device = torch.device("cuda")

        self.batcher = batcher
        self.config = config

        self.num_render_sequences = 2
        self.template_mesh = Mesh(filename=config['template_fname'])
        
        self.model = FaceFormerV2(self.config, self.device).to(self.device)

        self.optimizer = optim.Adam([p for p in self.model.parameters()], lr=1e-4)
        
        self.criterion = nn.MSELoss()
        # self.criterion = nn.SmoothL1Loss()

        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(osp.join(self.config['checkpoint_dir'], "logdir"))
        
        self.model_serializer = ModelSerializer(
            osp.join(self.config['checkpoint_dir'], "latest_model.ckpt"),
            osp.join(self.config['checkpoint_dir'], "best_model.ckpt"))
    
    def restore(self, load_latest=True, load_best=False):
        start_epoch, valid_loss_min = 1, 1000.0

        if load_best:
            start_epoch, valid_loss_min = self.model_serializer.load_ckp(
                osp.join(self.config['checkpoint_dir'], "best_model.ckpt"), self.model, self.optimizer)
        elif load_latest and osp.exists(osp.join(self.config['checkpoint_dir'], "latest_model.ckpt")):
            start_epoch, valid_loss_min = self.model_serializer.load_ckp(
                osp.join(self.config['checkpoint_dir'], "latest_model.ckpt"), self.model, self.optimizer)
            print(f"[INFO] Load latest checkpoint start_epoch: {start_epoch}")
        else:
            print("[WARNING] Train from scratch!")
        return start_epoch, valid_loss_min

    def train(self):
        num_train_batches = self.batcher.get_num_batches(self.config['batch_size']) + 1
        global_step = 0
        
        start_epoch, _ = self.restore(load_latest=True)

        for epoch in range(start_epoch, self.config['epoch_num'] + 1):
            for iter in range(num_train_batches):
                loss = self._training_step()
                
                print(f"Epoch: {epoch} | Iter: {iter} | Global Step: {global_step} | Loss: {loss}")
                self.tb_writer.add_scalar("traing_loss", loss.item(), global_step)
                # if iter % 100 == 0:
                #     val_loss = self._validation_step()
                #     logging.warning("Validation loss: %.6f" % val_loss)
                #     self.tb_writer.add_scalar("validation_loss", val_loss, global_step)
                
                global_step += 1

            if epoch % 2 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'valid_loss_min': 1000.0,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                self.model_serializer.save(checkpoint, False)
                print(f"Saving checkpoint in epoch {epoch}")

            if epoch % 4 == 0:
                self._render_sequences(out_folder=osp.join(self.config['checkpoint_dir'], 'videos', f'training_epoch_{epoch}_iter_{iter}'), 
                                       data_specifier='training')
            #     self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 
            #                                                    'validation_epoch_%d_iter_%d' % (epoch, iter)), data_specifier='validation')

        print("Training Done!")

    def _prepare_data(self, batch_data_dict, device, normalize=True):
        batch_size, seq_len = batch_data_dict['face_vertices'].shape[:2]

        #======= Prepare the GT face motion ==========#
        batch_data_dict['target_face_motion'] = \
            batch_data_dict['face_vertices'] - np.expand_dims(batch_data_dict['face_template'], axis=1) # (B, Sy, 5023, 3)

        if normalize: # convert to [0, 1]
            batch_data_dict['target_face_motion'] = normalize_motion(batch_data_dict['target_face_motion'])

        batch_data_dict['target_face_motion'] = \
            batch_data_dict['target_face_motion'].reshape(batch_size, seq_len, -1) # (B, Sy, 15069)

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
        batch_size, seq_len = pred_facial_motion.shape[:2]

        # pred_facial_motion = torch.reshape(pred_facial_motion, (batch_size, seq_len, 5023, 3))
        # print(torch.min(pred_facial_motion), torch.max(pred_facial_motion))
        # print(torch.min(batch_data_dict['target_face_motion']), torch.max(batch_data_dict['target_face_motion']))

        # pred_facial_vertices = batch_data_dict['face_template'].unsqueeze(1) + pred_facial_motion

        loss = self.criterion(pred_facial_motion, batch_data_dict['target_face_motion'])

        loss.backward()

        self.optimizer.step()

        return loss
    
    def _validation_step(self):
        self.model.eval()
    
    def _test_step(self, data_dict):
        """Assume the data dictionary is from one sequence

        Args:
            data_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.model.eval()

        ## slice the data_dict
        splited_face_vertices_list = split_given_size(data_dict['face_vertices'], 60)
        splited_face_raw_audio_list = split_given_size(data_dict['raw_audio'], 22000)

        data_dict['face_vertices'] = np.stack(splited_face_vertices_list[:-1])
        data_dict['raw_audio'] = np.stack(splited_face_raw_audio_list[:-1])
        data_dict['face_template'] = np.tile(data_dict['face_template'], (len(splited_face_vertices_list[:-1]), 1, 1))
        data_dict['subject_idx'] = data_dict['subject_idx'].repeat(len(splited_face_vertices_list[:-1]), axis=0) # (B, )

        self._prepare_data(data_dict, self.device)
        
        ## Network forward
        pred_facial_motion = self.model.inference(data_dict)

        batch_size, seq_len = pred_facial_motion.shape[:2]
        
        pred_facial_motion = torch.reshape(pred_facial_motion, (batch_size, seq_len, -1, 3))

        pred_facial_motion = denormalize_motion(pred_facial_motion)
        
        pred_facial_vertices = data_dict['face_template'].unsqueeze(1) + pred_facial_motion # (B, S, 5023, 3)

        ## Reshape to (B*S, 5023, 3)
        pred_facial_vertices = torch.reshape(pred_facial_vertices, (-1, 5023, 3))
        
        return pred_facial_vertices.cpu().detach().numpy()

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

    def _render_sequences(self, out_folder, run_in_parallel=True, data_specifier='validation'):
        print('Render %s sequences' % data_specifier)
        if run_in_parallel:
            thread = threading.Thread(target=self._render_helper, args=(out_folder, data_specifier))
            thread.start()
            thread.join()
        else:
            self._render_helper(out_folder, data_specifier)

    def _render_helper(self, out_folder, data_specifier):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if data_specifier == 'training':
            data_dict = self.batcher.get_training_sequences_in_order(self.num_render_sequences)
            #Render each training sequence with the corresponding condition
            subject_idx = data_dict['subject_idx']
            condition_subj_idx = [[idx] for idx in subject_idx]
        elif data_specifier == 'validation':
            data_dict = self.batcher.get_validation_sequences_in_order(
                self.num_render_sequences)
            #Render each validation sequence with all training conditions
            num_training_subjects = self.batcher.get_num_training_subjects()
            subject_idx = data_dict['subject_idx']
            condition_subj_idx = [range(num_training_subjects) for idx in subject_idx]
        else:
            raise NotImplementedError('Unknown data specifier %s' % data_specifier)

        for i_seq in range(len(data_dict['raw_audio'])):
            conditions = condition_subj_idx[i_seq]
            for condition_idx in conditions:
                condition_subj = self.batcher.convert_training_idx2subj(condition_idx)
                video_fname = os.path.join(out_folder, '%s_%03d_condition_%s.mp4' % (data_specifier, i_seq, condition_subj))
                self._render_sequences_helper(video_fname, 
                                              data_dict['raw_audio'][i_seq], 
                                              data_dict['face_template'][i_seq], 
                                              data_dict['face_vertices'][i_seq], 
                                              condition_idx)

    def _render_sequences_helper(self, video_fname, seq_raw_audio, seq_template, seq_verts, condition_idx):
        def add_image_text(img, text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) // 2
            textY = textsize[1] + 10
            cv2.putText(img, '%s' % (text), (textX, textY), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        wavfile.write(tmp_audio_file.name, 22000, seq_raw_audio)

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (1600, 800), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1600, 800), True)

        ## ============= Network forward =================== ##
        data_dict = {'raw_audio': seq_raw_audio, 
                     'face_vertices': seq_verts, 
                     'face_template': np.expand_dims(seq_template, axis=0), 
                     'subject_idx': np.array([condition_idx])}
        predicted_vertices = self._test_step(data_dict)

        center = np.mean(seq_verts[0], axis=0)

        num_frames = predicted_vertices.shape[0]
        for i_frame in range(num_frames):
            gt_img = render_mesh_helper(Mesh(seq_verts[i_frame], self.template_mesh.f), center)
            gt_img = np.ascontiguousarray(gt_img, dtype=np.uint8)
            add_image_text(gt_img, 'Captured data')

            pred_img = render_mesh_helper(Mesh(predicted_vertices[i_frame], self.template_mesh.f), center)
            pred_img = np.ascontiguousarray(pred_img, dtype=np.uint8)
            add_image_text(pred_img, 'VOCA prediction')
            img = np.hstack((gt_img, pred_img))
            writer.write(img)
        writer.release()

        cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            tmp_audio_file.name, tmp_video_file.name, video_fname)).split()
        call(cmd)

def main():
    from omegaconf import OmegaConf

    config = OmegaConf.load('./config/config.yaml')

    #========= Loading Dataset =========#
    batcher = get_dataset(config['dataset'])

    #========= Create Model ============#
    model = Trainer(config, batcher)
    model.train()


if __name__ == "__main__":
    main()
