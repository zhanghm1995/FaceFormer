'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-13 11:40:15
Email: haimingzhang@link.cuhk.edu.cn
Description: Test FaceFormer Encoder to process the audio input
'''

import numpy as np
import torch
import torchaudio
from wav2vec2 import FaceFormerEncoder, FaceFormer
import argparse
from omegaconf import OmegaConf

def preprocess_audio(waveform):
    waveform = torch.from_numpy(waveform).to(device)

    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    
    sample_rate = 22000
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform


def test_faceformer_encoder():
    device = torch.device("cuda")

    encoder = FaceFormerEncoder(device).to(device)

    input_audio = []
    raw_audio = np.load("60_FaceTalk_170728_03272_TA_sentence16_audio.npy").astype(np.float32)
    # input_audio.append(raw_audio)
    # raw_audio = np.load("294_FaceTalk_170904_03276_TA_sentence34_audio.npy").astype(np.float32)
    # input_audio.append(raw_audio)

    # raw_audio = np.stack(input_audio)

    waveform = preprocess_audio(raw_audio)
    print(raw_audio.shape, waveform.shape)

    input_dict = {"waveforms": waveform}
    output = encoder(input_dict)
    print(output.shape)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')
    parser.add_argument('--cfg', type=str, default='./config/config.yaml', help='the config_file')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')

    args = parser.parse_args()
    config = OmegaConf.load(args.cfg)
    config.update(vars(args)) # override the configuration using the value in args

    return config

def test_faceformer():
    device = torch.device("cuda")

    config = parse_config()

    model = FaceFormer(config, 15069, device).to(device)

    ## Create the input
    target_face_motion = torch.randn(64, 60, 5023, 3).to(device)
    subject_idx = torch.randn(64, 60, 8).to(device)
    raw_audio = torch.randn(64, 22000).to(device)

    data_dict = {"target_face_motion": target_face_motion,
                 "subject_idx": subject_idx,
                 "raw_audio": raw_audio}
    output = model(data_dict)
    
    
test_faceformer()
