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
from dataset.voca_dataset import DataHandler, Batcher
from test_voca_dataset import one_hot
from wav2vec2 import FaceFormer, FaceFormerV2
from utils.rendering import render_mesh_helper
from psbody.mesh import Mesh


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
            #     self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 
            #                                                    'training_epoch_%d_iter_%d' % (epoch, iter)), data_specifier='training')
            #     self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 
            #                                                    'validation_epoch_%d_iter_%d' % (epoch, iter)), data_specifier='validation')

    def _prepare_data(self, batch_data_dict, device):
        batch_size, seq_len = batch_data_dict['face_vertices'].shape[:2]

        #======= Prepare the GT face motion ==========#
        batch_data_dict['target_face_motion'] = \
            batch_data_dict['face_vertices'] - np.expand_dims(batch_data_dict['face_template'], axis=1)
        batch_data_dict['target_face_motion'] = batch_data_dict['target_face_motion'].reshape(batch_size, seq_len, -1)

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

        pred_facial_motion = torch.reshape(pred_facial_motion, (batch_size, seq_len, 5023, 3))

        pred_facial_vertices = batch_data_dict['face_template'].unsqueeze(1) + pred_facial_motion

        loss = self.criterion(pred_facial_vertices, batch_data_dict['face_vertices'])
        loss.backward()

        return loss
    
    def _validation_step(self):
        self.model.eval()
    
    def _test_step(self, data_dict):
        self.model.eval()
        pred_facial_motion = self.model.inference(data_dict)
        
        pred_facial_vertices = data_dict['face_template'].unsqueeze(1) + pred_facial_motion
        
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
            # self.threads.append(threading.Thread(target=self._render_helper, args=(out_folder, data_specifier)))
            # self.threads[-1].start()
            thread = threading.Thread(target=self._render_helper, args=(out_folder, data_specifier))
            thread.start()
            thread.join()
        else:
            self._render_helper(out_folder, data_specifier)

    def _render_helper(self, out_folder, data_specifier):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if data_specifier == 'training':
            data_dict = self.batcher.get_training_sequences_in_order(
                self.num_render_sequences)
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

        num_frames = seq_verts.shape[0]
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        wavfile.write(tmp_audio_file.name, seq_raw_audio['sample_rate'], seq_raw_audio['audio'])

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (1600, 800), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1600, 800), True)

        ## ============= Network forward =================== ##
        predicted_vertices = self._test_step({'raw_audio': seq_raw_audio, 'face_template': seq_template})
        predicted_vertices = torch.squeeze(predicted_vertices)
        
        center = np.mean(seq_verts[0], axis=0)

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
    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)

    #========= Create Model ============#
    model = Trainer(config, batcher)
    model.train()


if __name__ == "__main__":
    main()
