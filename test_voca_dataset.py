'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-12 17:29:02
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the VOCADataset class
'''
import os
import shutil
import numpy as np
import configparser
from dataset import get_dataset
from config_parser import read_config, create_default_config
import torch


def one_hot(x):
    """Get the one hot matrix

    Args:
        x (Tensor): Bxseq_len dimension

    Returns:
        [type]: Bxseq_lenx8
    """
    x = x.unsqueeze(-1)
    condition = torch.zeros(x.shape[0], x.shape[1], 8).scatter_(2, x.type(torch.LongTensor), 1)
    return condition

def _prepare_data(batch_data_dict):
        batch_size, seq_len = batch_data_dict['face_vertices'].shape[:2]

        #======= Prepare the GT face motion ==========#
        batch_data_dict['target_face_motion'] = \
            batch_data_dict['face_vertices'] - np.expand_dims(batch_data_dict['face_template'], axis=1)

        #======== Prepare the subject idx ===========#
        subject_idx = np.expand_dims(np.stack(batch_data_dict['subject_idx']), -1)
        batch_data_dict['subject_idx'] = one_hot(torch.from_numpy(subject_idx.repeat(seq_len, axis=-1))).to(torch.float32)


def test_voca_dataset(config):
    batcher = get_dataset(config)

    batch_data_dict = batcher.get_training_batch(config['batch_size'])

    _prepare_data(batch_data_dict)

    for key, value in batch_data_dict.items():
        if key != "subject_idx":
            batch_data_dict[key] = torch.from_numpy(value).type(torch.FloatTensor)

    for key, value in batch_data_dict.items():
        print(key, value.shape, value.dtype)

    # audio = processed_audio[0]['audio']
    # num_face_frames = face_vertices[0].shape[0]
    # print(audio.shape, num_face_frames, seq_info[0])
    # np.save(f"{num_face_frames}_{seq_info[0][0]}_{seq_info[0][1]}_audio.npy", audio)

def test_voca_dataset_get_sequences(config):
    def split_given_size(a, size):
        return np.split(a, np.arange(size, len(a), size))

    batcher = get_dataset(config)

    data_dict = batcher.get_training_sequences_in_order(2)
    for key, value in data_dict.items():
        print(key, value[0].shape)
    
    splited_face_vertices_list = split_given_size(data_dict['face_vertices'][0], 60)
    splited_face_raw_audio = split_given_size(data_dict['raw_audio'][0], 22000)
    print(len(splited_face_vertices_list), len(splited_face_raw_audio))

    print(splited_face_vertices_list[-2].shape, splited_face_raw_audio[-2].shape)
    print(splited_face_vertices_list[-1].shape, splited_face_raw_audio[-1].shape)


def test_wav2vec2():
    import torch
    import torch.nn as nn
    import torchaudio
    import torchaudio.models.wav2vec2 as ta_wav2vec2

    device = torch.device("cuda")

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec2_model = bundle.get_model().to(device)
    print(wav2vec2_model.__class__)
    print("hhh")

    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


def main():
    pkg_path, _ = os.path.split(os.path.realpath(__file__))
    init_config_fname = os.path.join(pkg_path, 'training_config.cfg')
    if not os.path.exists(init_config_fname):
        print('Config not found %s' % init_config_fname)
        create_default_config(init_config_fname)

    config = configparser.RawConfigParser()
    config.read(init_config_fname)

    # Path to cache the processed audio
    config.set('Input Output', 'processed_audio_path',
               './training_data/processed_audio_%s.pkl' % config.get('Audio Parameters', 'audio_feature_type'))

    checkpoint_dir = config.get('Input Output', 'checkpoint_dir')
    if os.path.exists(checkpoint_dir):
        print('Checkpoint dir already exists %s' % checkpoint_dir)
        key = input('Press "q" to quit, "x" to erase existing folder, and any other key to continue training: ')
        if key.lower() == 'q':
            return
        elif key.lower() == 'x':
            try:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            except:
                print('Failed deleting checkpoint directory')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config_fname = os.path.join(checkpoint_dir, 'config.pkl')
    if os.path.exists(config_fname):
        print('Use existing config %s' % config_fname)
    else:
        with open(config_fname, 'w') as fp:
            config.write(fp)
            fp.close()

    config = read_config(config_fname)

    test_voca_dataset(config)


def main2():
    from omegaconf import OmegaConf

    config = OmegaConf.load('./config/config.yaml')
    test_voca_dataset(config['dataset'])
    # test_voca_dataset_get_sequences(config['dataset'])


if __name__ == "__main__":
    main2()

    # test_wav2vec2()

