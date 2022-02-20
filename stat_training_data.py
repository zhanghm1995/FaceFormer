'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-19 12:52:34
Email: haimingzhang@link.cuhk.edu.cn
Description: Compute the statistics of the training data
'''

from matplotlib.pyplot import axis
from tqdm import tqdm
import numpy as np
import pickle
from dataset.voca_dataset import DataHandler, load_from_config, invert_data2array


def stat_max_sequence_length(config):
    face_verts_mmaps_path = load_from_config(config, 'verts_mmaps_path')
    face_templates_path = load_from_config(config, 'templates_path')
    raw_audio_path = load_from_config(config, 'raw_audio_path')
    data2array_verts_path = load_from_config(config, 'data2array_verts_path')

    face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r') # (N, 5023, 3)
    templates_data = pickle.load(open(face_templates_path, 'rb'), encoding='latin1')
    raw_audio = pickle.load(open(raw_audio_path, 'rb'), encoding='latin1')

    data2array_verts = pickle.load(open(data2array_verts_path, 'rb'))

    num_frames_list = []
    for subj, value in tqdm(data2array_verts.items()):
        for seq, index in value.items():
            curr_seq_indices = list(index.values())
            num_frames_list.append(len(curr_seq_indices))
    print(len(num_frames_list), max(num_frames_list))
            

def stat_voca_training_data(config):
    """Statistic the VOCA training data to get 
       the minimum and maximum facial vertices value

    Args:
        config (dict): config parameters
    """
    face_verts_mmaps_path = load_from_config(config, 'verts_mmaps_path')
    face_templates_path = load_from_config(config, 'templates_path')
    raw_audio_path = load_from_config(config, 'raw_audio_path')
    data2array_verts_path = load_from_config(config, 'data2array_verts_path')

    face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r') # (N, 5023, 3)
    templates_data = pickle.load(open(face_templates_path, 'rb'), encoding='latin1')
    raw_audio = pickle.load(open(raw_audio_path, 'rb'), encoding='latin1')

    data2array_verts = pickle.load(open(data2array_verts_path, 'rb'))
    array2data_verts = invert_data2array(data2array_verts)

    min_all_frames_motion, max_all_frames_motion = [], []
    for subj, value in tqdm(data2array_verts.items()):
        curr_subj_template = templates_data[subj] # (5023, 3)

        for seq, index in value.items():
            curr_seq_indices = list(index.values())
            start_idx, end_idx = min(curr_seq_indices), max(curr_seq_indices) + 1
            curr_seq_vertices = face_vert_mmap[start_idx:end_idx, ...] # (N, 5023, 3)
            curr_seq_vertiecs_motion = curr_seq_vertices - np.expand_dims(curr_subj_template, axis=0)
            
            curr_seq_vertiecs_motion = np.reshape(curr_seq_vertiecs_motion, (-1, 3))

            min_all_frames_motion.append(np.min(curr_seq_vertiecs_motion, axis=0, keepdims=True))
            max_all_frames_motion.append(np.max(curr_seq_vertiecs_motion, axis=0, keepdims=True))
    
    min_all_frames_motion = np.concatenate(min_all_frames_motion, axis=0)
    max_all_frames_motion = np.concatenate(max_all_frames_motion, axis=0)
    print(min_all_frames_motion.shape, max_all_frames_motion.shape)

    min_value = np.min(min_all_frames_motion, axis=0)
    max_value = np.max(max_all_frames_motion, axis=0)
    print(min_value, max_value)


MIN_MOTION = [-0.00916614, -0.02674509, -0.0166305]
MAX_MOTION = [0.01042878, 0.01583716, 0.01325295]


def test_normalize(config):
    face_verts_mmaps_path = load_from_config(config, 'verts_mmaps_path')
    face_templates_path = load_from_config(config, 'templates_path')
    raw_audio_path = load_from_config(config, 'raw_audio_path')
    data2array_verts_path = load_from_config(config, 'data2array_verts_path')

    face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r') # (N, 5023, 3)
    templates_data = pickle.load(open(face_templates_path, 'rb'), encoding='latin1')
    raw_audio = pickle.load(open(raw_audio_path, 'rb'), encoding='latin1')

    data2array_verts = pickle.load(open(data2array_verts_path, 'rb'))
    array2data_verts = invert_data2array(data2array_verts)
    
    DIFF_MOTION = np.array(MAX_MOTION) - np.array(MIN_MOTION)

    for subj, value in tqdm(data2array_verts.items()):
        curr_subj_template = templates_data[subj] # (5023, 3)

        for seq, index in value.items():
            curr_seq_indices = sorted(index.values())
            start_idx, end_idx = min(curr_seq_indices), max(curr_seq_indices) + 1
            curr_seq_vertices = face_vert_mmap[start_idx:end_idx, ...] # (N, 5023, 3)
            curr_seq_vertiecs_motion = curr_seq_vertices - np.expand_dims(curr_subj_template, axis=0)
            
            curr_seq_vertiecs_motion = curr_seq_vertiecs_motion - np.expand_dims(np.expand_dims(np.array(MIN_MOTION), axis=0), axis=0)
            curr_seq_vertiecs_motion = curr_seq_vertiecs_motion / DIFF_MOTION[None, None, :]

            print(np.min(curr_seq_vertiecs_motion), np.max(curr_seq_vertiecs_motion))

    


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load('./config/config.yaml')

    stat_max_sequence_length(config['dataset'])
    exit(0)

    test_normalize(config['dataset'])

    # stat_voca_training_data(config['dataset'])

