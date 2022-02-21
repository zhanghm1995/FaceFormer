'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-13 15:26:22
Email: haimingzhang@link.cuhk.edu.cn
Description: script to run training pipeline
'''
import os
import os.path as osp
from sys import stderr
import cv2
import tempfile
import threading
from matplotlib.pyplot import axis
import numpy as np
import logging
from scipy.io import wavfile
import torch
import torch.nn as nn
from torch import optim
import subprocess
from subprocess import call
from tqdm import tqdm
from omegaconf import OmegaConf
from config_parser import get_configs
from dataset import get_dataset
from test_voca_dataset import one_hot
from wav2vec2 import FaceFormer, FaceFormerV2
from utils.rendering import render_mesh_helper
from psbody.mesh import Mesh
from utils.model_serializer import ModelSerializer
from utils.pytorch3d_renderer import FaceRenderer
from utils.loss import WeightedLoss


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
        
        self.criterion = nn.MSELoss(reduction='sum')
        # self.criterion = WeightedLoss()
        # self.criterion = nn.SmoothL1Loss()
        
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
        
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(osp.join(self.config['checkpoint_dir'], "logdir"))

        num_train_batches = self.batcher.get_num_batches(self.config['batch_size']) + 1
        
        start_epoch, _ = self.restore(load_latest=True)

        global_step = 0
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

            if epoch % 30 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'global_step': global_step + 1,
                    'valid_loss_min': 1000.0,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }   
                self.model_serializer.save(checkpoint, False)
                print(f"Saving checkpoint in epoch {epoch}")

            # if epoch % 1 == 0:
            #     self._render_sequences(out_folder=osp.join(self.config['checkpoint_dir'], 'videos', f'training_epoch_{epoch}_iter_{iter}'), 
            #                            data_specifier='training')
            #     self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 
            #                                                    'validation_epoch_%d_iter_%d' % (epoch, iter)), data_specifier='validation')

        print("Training Done!")
    
    def test(self):
        print("************ Start test ***************")
        def denormalize(input):
            batch_size, seq_len = input.shape[:2]
            input = torch.reshape(input, (batch_size, seq_len, -1, 3)).cpu().detach().numpy()
            output = denormalize_motion(input)
            return output

        from utils.save_data import save_video

        start_epoch, _ = self.restore(load_latest=True)

        self.model.eval()

        ## prepare datas
        batch_data_dict = self.batcher.get_training_batch(self.config['batch_size'])
        raw_audio = batch_data_dict['raw_audio'][0].cpu().numpy()

        self._prepare_data(batch_data_dict, self.device)

        ## forward
        pred_facial_motion = self.model.inference(batch_data_dict)
        loss = self.criterion(pred_facial_motion, batch_data_dict['target_face_motion'])

        diff = pred_facial_motion - batch_data_dict['target_face_motion']

        print(f"[INFO] Test Loss: {loss.item()}", torch.min(diff), torch.max(diff))

        pred_facial_motion = denormalize(pred_facial_motion)
        target_facial_motion = denormalize(batch_data_dict['target_face_motion'])
        pred_facial_vertices = batch_data_dict['face_template'].cpu().numpy() + pred_facial_motion[0] # (S, 5023, 3)

        # face_vertex = batch_data_dict['face_vertices'][0].type(torch.FloatTensor)
        face_vertex = torch.from_numpy(pred_facial_vertices).type(torch.FloatTensor)

        ## =============== Start rendering ======================== ##
        face_renderer = FaceRenderer(self.device, rendered_image_size=512)

        rendered_image = []
        for i in tqdm(range(0, len(face_vertex), 100)):
            sub_face_vertex = face_vertex[i:i+100].to(self.device)
            image_array = face_renderer(sub_face_vertex)
            rendered_image.append(image_array)

        rendered_image = np.concatenate(rendered_image, axis=0)
        print(rendered_image.shape)

        output_video_fname = osp.join("./output", f"FaceTalk_170728_03272_TA_seq01_infer_wo_padding_all_data.mp4")
        
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=osp.dirname(output_video_fname))
        wavfile.write(tmp_audio_file.name, 22000, raw_audio)

        save_video(rendered_image, output_video_fname, image_size=512, audio_fname=tmp_audio_file.name)

    def debug(self):
        self.model.train()
        ## prepare datas
        batch_data_dict = self.batcher.get_training_batch(self.config['batch_size'])
        self._prepare_data(batch_data_dict, self.device)

        pred_facial_motion = self.model(batch_data_dict)

    def _prepare_data(self, batch_data_dict, device, normalize=True):
        batch_size, seq_len = batch_data_dict['face_vertices'].shape[:2]

        #======= Prepare the GT face motion ==========#
        if normalize: # convert to [0, 1]
            batch_data_dict['target_face_motion'] = normalize_motion(batch_data_dict['target_face_motion'])

        batch_data_dict['target_face_motion'] = \
            batch_data_dict['target_face_motion'].reshape(batch_size, seq_len, -1) # (B, Sy, 15069)

        #======== Prepare the subject idx ===========#
        subject_idx = np.expand_dims(np.stack(batch_data_dict['subject_idx']), -1)
        batch_data_dict['subject_idx'] = one_hot(torch.from_numpy(subject_idx.repeat(seq_len, axis=-1))).to(torch.float32)

        for key, value in batch_data_dict.items():
            if key in ["raw_audio", "face_vertices", "target_face_motion"]:
                batch_data_dict[key] = value.to(device=device, dtype=torch.float32)
                continue
            if not torch.is_tensor(value):
                batch_data_dict[key] = torch.from_numpy(value).to(device)
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

        # loss = self.criterion(pred_facial_motion, batch_data_dict['target_face_motion'])
        loss = self.criterion(pred_facial_motion, batch_data_dict['target_face_motion']) / 100.0

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
        self._prepare_data(data_dict, self.device)
        
        ## Network forward
        pred_facial_motion = self.model.inference_whole_sequence(data_dict, 60, self.device) # (1, L, 15069)

        loss = self.criterion(pred_facial_motion, data_dict['target_face_motion'])
        print(f"[INFO] Test Loss: {loss.item()}")

        batch_size, seq_len = pred_facial_motion.shape[:2]
        
        pred_facial_motion = torch.reshape(pred_facial_motion, (batch_size, seq_len, -1, 3)).cpu().detach().numpy()

        pred_facial_motion = denormalize_motion(pred_facial_motion)
        
        pred_facial_vertices = data_dict['face_template'].cpu().numpy() + pred_facial_motion[0] # (S, 5023, 3)

        return pred_facial_vertices

    def _test_step_old(self, data_dict):
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

        loss = self.criterion(pred_facial_motion, data_dict['target_face_motion'])
        print(f"[INFO] Test Loss: {loss.item()}")

        batch_size, seq_len = pred_facial_motion.shape[:2]
        
        pred_facial_motion = torch.reshape(pred_facial_motion, (batch_size, seq_len, -1, 3)).cpu().detach().numpy()

        pred_facial_motion = denormalize_motion(pred_facial_motion)
        
        pred_facial_vertices = data_dict['face_template'].unsqueeze(1).cpu().detach().numpy() + pred_facial_motion # (B, S, 5023, 3)

        ## Reshape to (B*S, 5023, 3)
        pred_facial_vertices = np.reshape(pred_facial_vertices, (-1, 5023, 3))
        
        return pred_facial_vertices

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
            data_list = self.batcher.get_training_sequences_in_order(self.num_render_sequences)
            # Render each training sequence with the corresponding condition
            condition_subj_idx = [[data['subject_idx']] for data in data_list]
        elif data_specifier == 'validation':
            data_dict = self.batcher.get_validation_sequences_in_order(
                self.num_render_sequences)
            #Render each validation sequence with all training conditions
            num_training_subjects = self.batcher.get_num_training_subjects()
            subject_idx = data_dict['subject_idx']
            condition_subj_idx = [range(num_training_subjects) for idx in subject_idx]
        else:
            raise NotImplementedError('Unknown data specifier %s' % data_specifier)

        for i_seq, seq_dict in enumerate(data_list):
            conditions = condition_subj_idx[i_seq]
            for condition_idx in conditions:
                condition_subj = self.batcher.convert_training_idx2subj(condition_idx)
                video_fname = os.path.join(out_folder, '%s_%03d_condition_%s.mp4' % (data_specifier, i_seq, condition_subj))
                self._render_sequences_helper(video_fname, 
                                              seq_dict, 
                                              condition_idx)

    def _render_sequences_helper(self, video_fname, data_dict, condition_idx):
        def add_image_text(img, text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) // 2
            textY = textsize[1] + 10
            cv2.putText(img, '%s' % (text), (textX, textY), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        wavfile.write(tmp_audio_file.name, 22000, data_dict['raw_audio'])

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1600, 800), True)

        ## ============= Network forward =================== ##
        input_data_dict = {'raw_audio': np.expand_dims(data_dict['raw_audio'], axis=0),
                           'face_vertices': np.expand_dims(data_dict['face_vertices'], axis=0),
                           'face_template': np.expand_dims(data_dict['face_template'], axis=0), 
                           'subject_idx': np.expand_dims(np.array(condition_idx), axis=0)}
        predicted_vertices = self._test_step(input_data_dict)

        center = np.mean(data_dict['face_vertices'][0], axis=0)

        num_frames = predicted_vertices.shape[0]
        for i_frame in range(num_frames):
            gt_img = render_mesh_helper(Mesh(data_dict['face_vertices'][i_frame], self.template_mesh.f), center)
            gt_img = np.ascontiguousarray(gt_img, dtype=np.uint8)
            add_image_text(gt_img, 'Captured data')

            pred_img = render_mesh_helper(Mesh(predicted_vertices[i_frame], self.template_mesh.f), center)
            pred_img = np.ascontiguousarray(pred_img, dtype=np.uint8)
            add_image_text(pred_img, 'VOCA prediction')
            img = np.hstack((gt_img, pred_img))
            writer.write(img)
        writer.release()

        cmd = f'ffmpeg -i {tmp_audio_file.name} -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {video_fname}'
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    #========= Loading Config =========#
    config = OmegaConf.load('./config/config.yaml')

    #========= Loading Dataset =========#
    batcher = get_dataset(config['dataset'])

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    OmegaConf.save(config, osp.join(config['checkpoint_dir'], "config.yaml"))

    #========= Create Model ============#
    model = Trainer(config, batcher)
    model.train()


if __name__ == "__main__":
    main()
