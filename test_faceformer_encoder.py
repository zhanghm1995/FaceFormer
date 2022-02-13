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
from wav2vec2 import FaceFormerEncoder


def preprocess_audio(waveform):
    waveform = torch.from_numpy(waveform).to(device)

    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    
    sample_rate = 22000
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform

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