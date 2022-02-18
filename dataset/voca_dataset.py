'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.
Author: Haiming Zhang
Date: 2022-02-11 15:09:27
Email: haimingzhang@link.cuhk.edu.cn
Description: Load VOCASET dataset, adapted from voca-pytorch repo
'''

import os
from typing import List
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm


def get_sub_list_randomly(input_list, sub_list_len):
    assert sub_list_len <= len(input_list)
    
    len_diff = len(input_list) - sub_list_len
    
    start_idx = random.randint(0, len_diff)
    start_idx = max(0, start_idx - 30)
    return input_list[start_idx:start_idx+sub_list_len], start_idx


def load_from_config(config, key):
    if key in config:
        return config[key]
    else:
        raise ValueError('Key does not exist in config %s' % key)

def invert_data2array(data2array):
    array2data = {}
    for sub in data2array.keys():
        for seq in data2array[sub].keys():
            for frame, array_idx in data2array[sub][seq].items():
                array2data[array_idx] = (sub, seq, frame)
    return array2data

def compute_window_array_idx(data2array, window_size):
    def window_frame(frame_idx, window_size):
        l0 = max(frame_idx + 1 - window_size, 0)
        l1 = frame_idx + 1
        window_frames = np.zeros(window_size, dtype=int)
        window_frames[window_size - l1 + l0:] = np.arange(l0, l1)
        return window_frames

    array2window_ids = {}
    for sub in data2array.keys():
        for seq in data2array[sub].keys():
            for frame, array_idx in data2array[sub][seq].items():
                window_frames = window_frame(frame, window_size)
                array2window_ids[array_idx] = [data2array[sub][seq][id] for id in window_frames]
    return array2window_ids


class DataHandler:
    """Class to load VOCA training dataset
    """
    def __init__(self, config, for_faceformer=True):
        self.training_subjects = config['subject_for_training'].split(" ")
        self.training_sequences = config['sequence_for_training'].split(" ")
        self.validation_subjects = config['subject_for_validation'].split(" ")
        self.validation_sequences = config['sequence_for_validation'].split(" ")
        self.testing_subjects = config['subject_for_testing'].split(" ")
        self.testing_sequences = config['sequence_for_testing'].split(" ")

        self.num_training_subjects = len(self.training_subjects)
        self.sequence_length = config['sequence_length']

        print("===================== Start loading data =====================")
        self._load_data(config)
        self._get_all_sequences_list()
        
        print("Initialize training, validation, and test indices")
        self._init_indices()

    def _get_all_sequences_list(self):
        """Get all training, validation and testing sequences
        """
        def get_sequences_info(subjects_list, sequences_list):
            sub_seq_list = []
            for subj in subjects_list:
                if subj not in self.data2array_verts:
                    continue
                for seq in sequences_list:
                    if(seq not in self.raw_audio[subj]) or (seq not in self.data2array_verts[subj]):
                        continue
                    sub_seq_list.append((subj, seq))
            return sub_seq_list
        
        self.all_training_sequences = get_sequences_info(
            self.training_subjects, self.training_sequences)
        self.all_validation_sequences = get_sequences_info(
            self.validation_subjects, self.validation_sequences)
        self.all_testing_sequences = get_sequences_info(
            self.testing_subjects, self.testing_sequences)
        print(f"number of all training sequence: {len(self.all_training_sequences)}, "
              f"number of all validation sequence: {len(self.all_validation_sequences)}",
              f"number of all testing sequence: {len(self.all_testing_sequences)}")

    def _init_indices(self):
        """Initialize the indices for FaceFormer training
        """
        self.training_idx2subj = {idx: self.training_subjects[idx] for idx in np.arange(len(self.training_subjects))}
        self.training_subj2idx = {self.training_idx2subj[idx]: idx for idx in self.training_idx2subj.keys()}

        def get_indices(input_sequences):
            output = []
            for item in tqdm(input_sequences):
                subj, seq = item

                raw_audio = self.raw_audio[subj][seq] # dictionary
                frame_array_indices = list(self.data2array_verts[subj][seq].values())

                num_data_frames = len(frame_array_indices)

                audio_sample_rate = raw_audio['sample_rate']

                raw_audio_length = int(self.sequence_length * audio_sample_rate / 60)

                for i in range(0, num_data_frames, self.sequence_length):
                    start_idx, end_idx = i, i + self.sequence_length
                    audio_start_idx = round(start_idx / 60.0 * audio_sample_rate)
                    audio_end_idx = round(end_idx / 60.0 * audio_sample_rate)

                    curr_raw_audio = raw_audio['audio'][audio_start_idx:audio_end_idx]
                    curr_frame_indices = frame_array_indices[start_idx:end_idx]

                    if len(curr_frame_indices) != self.sequence_length or curr_raw_audio.shape[0] != raw_audio_length:
                        continue
                    
                    sequence_data_dict = {}
                    sequence_data_dict['face_vertices'] = self.face_vert_mmap[curr_frame_indices]
                    sequence_data_dict['face_template'] = self.templates_data[subj]
                    sequence_data_dict['subject_idx'] = self.convert_training_subj2idx(subj)
                    sequence_data_dict['raw_audio'] = curr_raw_audio
                    output.append(sequence_data_dict)
            
            return output

        self.training_indices = get_indices(self.all_training_sequences)
        self.validation_indices = get_indices(self.all_validation_sequences)
        self.testing_indices = get_indices(self.all_testing_sequences)
        
    def get_data_splits(self):
        return self.training_indices, self.validation_indices, self.testing_indices

    def slice_data(self, indices):
        return self._slice_data_helper(indices)

    def get_training_sequences(self, num_sequences):
        return self._get_random_sequences(self.training_subjects, self.training_sequences, num_sequences)

    def get_validation_sequences(self, num_sequences):
        return self._get_random_sequences(self.validation_subjects, self.validation_sequences, num_sequences)

    def get_testing_sequences(self, num_sequences):
        return self._get_random_sequences(self.testing_subjects, self.testing_sequences, num_sequences)

    def get_num_training_subjects(self):
        return len(self.training_subjects)

    def convert_training_idx2subj(self, idx):
        if idx in self.training_idx2subj:
            return self.training_idx2subj[idx]
        else:
            return -1

    def convert_training_subj2idx(self, subj):
        if subj in self.training_subj2idx:
            return self.training_subj2idx[subj]
        else:
            return -1

    def _slice_data(self, indices):
        if self.num_consecutive_frames == 1:
            return self._slice_data_helper(indices)
        else:
            window_indices = []
            for id in indices:
                window_indices += self.array2window_ids[id]
            return self._slice_data_helper(window_indices)

    def _slice_data_helper(self, indices):
        raw_audio = []
        face_vertices = []
        face_templates = []
        subject_idx = []
        for item in indices:
            raw_audio.append(item['raw_audio'])
            face_templates.append(item['face_template'])
            face_vertices.append(item['face_vertices'])
            subject_idx.append(item['subject_idx'])

        batch_data_dict = {}
        batch_data_dict['face_vertices'] = np.stack(face_vertices) # (B, N, 5023, 3)
        batch_data_dict['raw_audio'] = np.stack(raw_audio) # (B, 22000)
        batch_data_dict['face_template'] = np.stack(face_templates) # (B, 5023, 3)
        batch_data_dict['subject_idx'] = np.stack(subject_idx) # (B, )

        return batch_data_dict

    def _slice_data_helper_v2(self, indices):
        batched_raw_audio, batched_face_vertices = [], []
        batched_face_template, batched_subject_idx = [], []
        batched_seq_info = []

        sequence_length = 60 # 1s

        for item in indices:
            subj, seq = item
            
            frame_array_indices = list(self.data2array_verts[subj][seq].values())

            frame_array_indices, start_idx = get_sub_list_randomly(
                frame_array_indices, sequence_length)

            curr_seq_face_vertices = self.face_vert_mmap[frame_array_indices]
            curr_seq_face_template = self.templates_data[subj]
            curr_subject_idx = self.convert_training_subj2idx(subj)

            audio_start_idx = round(22000 / 60.0 * (start_idx + 1)) - 1
            audio_end_idx = round(22000 / 60.0 * (start_idx + sequence_length + 1)) - 1

            curr_raw_audio = self.raw_audio[subj][seq]['audio'][audio_start_idx:audio_end_idx]

            if curr_raw_audio.shape[0] != 22000:
                print("====", subj, seq, audio_start_idx, audio_end_idx, self.raw_audio[subj][seq]['audio'].shape)

            batched_face_vertices.append(curr_seq_face_vertices)
            batched_face_template.append(curr_seq_face_template)
            batched_subject_idx.append(curr_subject_idx)
            batched_raw_audio.append(curr_raw_audio)
            batched_seq_info.append(item)
        
        batch_data_dict = {}
        batch_data_dict['face_vertices'] = np.stack(batched_face_vertices) # (B, N, 5023, 3)
        batch_data_dict['face_template'] = np.stack(batched_face_template) # (B, 5023, 3)
        batch_data_dict['subject_idx'] = np.stack(batched_subject_idx) # (B, )
        try:
            batch_data_dict['raw_audio'] = np.stack(batched_raw_audio)
        except:
            all_shape = [s.shape for s in batched_raw_audio]
            print(all_shape)
 
        return batch_data_dict

    def _load_data(self, config):
        face_verts_mmaps_path = load_from_config(config, 'verts_mmaps_path')
        face_templates_path = load_from_config(config, 'templates_path')
        raw_audio_path = load_from_config(config, 'raw_audio_path')
        data2array_verts_path = load_from_config(config, 'data2array_verts_path')

        self.face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r')
        self.templates_data = pickle.load(open(face_templates_path, 'rb'), encoding='latin1')
        self.raw_audio = pickle.load(open(raw_audio_path, 'rb'), encoding='latin1')

        self.data2array_verts = pickle.load(open(data2array_verts_path, 'rb'))
        self.array2data_verts = invert_data2array(self.data2array_verts)

    def _init_data_splits(self, subject_for_training: List, sequence_for_training, subject_for_validation,
                          sequence_for_validation, subject_for_testing, sequence_for_testing):
        def select_valid_subjects(subjects_list):
            return [subj for subj in subjects_list]

        def select_valid_sequences(sequences_list):
            return [seq for seq in sequences_list]

        self.training_subjects = select_valid_subjects(subject_for_training)
        self.training_sequences = select_valid_sequences(sequence_for_training)

        self.validation_subjects = select_valid_subjects(subject_for_validation)
        self.validation_sequences = select_valid_sequences(sequence_for_validation)

        self.testing_subjects = select_valid_subjects(subject_for_testing)
        self.testing_sequences = select_valid_sequences(sequence_for_testing)

        all_instances = []
        for i in self.training_subjects:
            for j in self.training_sequences:
                all_instances.append((i, j))
        for i in self.validation_subjects:
            for j in self.validation_sequences:
                all_instances.append((i, j))
        for i in self.testing_subjects:
            for j in self.testing_sequences:
                all_instances.append((i, j))

        # All instances should contain all unique elements, otherwise the arguments were passed wrongly, so assertion
        if len(all_instances) != len(set(all_instances)):
            raise ValueError('User-specified data split not disjoint')

    def _get_random_sequences(self, subjects, sequences, num_sequences):
        if num_sequences == 0:
            return

        sub_seq_list = []
        for subj in subjects:
            if subj not in self.data2array_verts:
                continue
            for seq in sequences:
                if(seq not in self.raw_audio[subj]) or (seq not in self.data2array_verts[subj]):
                    continue
                sub_seq_list.append((subj, seq))
        st = random.getstate()
        random.seed(777)
        random.shuffle(sub_seq_list)
        random.setstate(st)

        if num_sequences > 0 and num_sequences < len(sub_seq_list):
            sub_seq_list = sub_seq_list[:num_sequences]
        return self._get_subject_sequences(sub_seq_list)

    def _get_subject_sequences(self, subject_sequence_list):
        face_vertices = []
        face_templates = []
        subject_idx = []

        raw_audio = []
        processed_audio = []
        for subj, seq in subject_sequence_list:
            frame_array_indices = []
            try:
                for frame, array_idx in self.data2array_verts[subj][seq].items():
                    frame_array_indices.append(array_idx)
            except KeyError:
                continue
            
            face_vertices.append(self.face_vert_mmap[frame_array_indices])
            face_templates.append(self.templates_data[subj])
            subject_idx.append(self.convert_training_subj2idx(subj))
            raw_audio.append(self.raw_audio[subj][seq]['audio'])
            processed_seq_audio = []
            if self.processed_audio is not None:
                for frame, array_idx in self.data2array_verts[subj][seq].items():
                    processed_seq_audio.append(self.processed_audio[subj][seq]['audio'][frame])
            processed_audio.append(processed_seq_audio)
        
        data_dict = {}
        data_dict['face_vertices'] = face_vertices
        data_dict['face_template'] = face_templates
        data_dict['subject_idx'] = subject_idx
        data_dict['raw_audio'] = raw_audio
 
        return data_dict
