'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.
Author: Haiming Zhang
Date: 2022-02-11 15:09:27
Email: haimingzhang@link.cuhk.edu.cn
Description: Load VOCASET dataset, adapted from voca-pytorch repo
'''

import os
import copy
import pickle
import random
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


def get_sub_list_randomly(input_list, sub_list_len):
    assert sub_list_len <= len(input_list)
    
    len_diff = len(input_list) - sub_list_len
    
    start_idx = random.randint(0, len_diff)
    start_idx = max(0, start_idx - 30)
    return input_list[start_idx:start_idx+sub_list_len], start_idx


class Batcher(Dataset):
    def __init__(self, data_handler):
        self.data_handler = data_handler

        data_splits = data_handler.get_data_splits()
        self.training_indices = copy.deepcopy(data_splits[0])
        self.val_indices = copy.deepcopy(data_splits[1])
        self.test_indices = copy.deepcopy(data_splits[2])

        self.current_state = 0

    def __len__(self):
        return len(self.training_indices)

    def get_training_size(self):
        return len(self.training_indices)

    def get_num_training_subjects(self):
        return self.data_handler.get_num_training_subjects()

    def convert_training_idx2subj(self, idx):
        return self.data_handler.convert_training_idx2subj(idx)

    def convert_training_subj2idx(self, subj):
        return self.data_handler.convert_training_subj2idx(subj)

    def get_training_batch(self, batch_size):
        """
        Get batch for training, main interface function
        :param batch_size:
        :return:
        """
        if self.current_state == 0:
            random.shuffle(self.training_indices)

        if (self.current_state + batch_size) > (len(self.training_indices) + 1):
            self.current_state = 0
            return self.get_training_batch(batch_size)
        else:
            self.current_state += batch_size
            batch_indices = self.training_indices[self.current_state:(self.current_state + batch_size)]
            if len(batch_indices) != batch_size:
                self.current_state = 0
                return self.get_training_batch(batch_size)
            return self.data_handler.slice_data(batch_indices)

    def get_validation_batch(self, batch_size):
        """
        Validation batch for randomize, quantitative evaluation
        :param batch_size:
        :return:
        """
        if batch_size > len(self.val_indices):
            return self.data_handler.slice_data(self.val_indices)
        else:
            return self.data_handler.slice_data(list(np.random.choice(self.val_indices, size=batch_size)))

    def get_test_batch(self, batch_size):
        if batch_size > len(self.test_indices):
            return self.data_handler.slice_data(self.test_indices)
        else:
            return self.data_handler.slice_data(list(np.random.choice(self.test_indices, size=batch_size)))

    def get_num_batches(self, batch_size):
        return int(float(len(self.training_indices)) / float(batch_size))

    def get_training_sequences_in_order(self, num_of_sequences):
        return self.data_handler.get_training_sequences(num_of_sequences)

    def get_validation_sequences_in_order(self, num_of_sequences):
        return self.data_handler.get_validation_sequences(num_of_sequences)

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
    def __init__(self, config, for_faceformer=True):
        subject_for_training = config['subject_for_training'].split(" ")
        sequence_for_training = config['sequence_for_training'].split(" ")
        subject_for_validation = config['subject_for_validation'].split(" ")
        sequence_for_validation = config['sequence_for_validation'].split(" ")
        subject_for_testing = config['subject_for_testing'].split(" ")
        sequence_for_testing = config['sequence_for_testing'].split(" ")
        self.num_consecutive_frames = config['num_consecutive_frames']
        self.audio_path = './training_data/processed_audio_deepspeech.pkl'

        print("Loading data")
        self._load_data(config)
        print("Initialize data splits")
        self._init_data_splits(subject_for_training, sequence_for_training, subject_for_validation,
                               sequence_for_validation, subject_for_testing, sequence_for_testing)
        print("Initilize all sequences")
        self._get_all_sequences_list()
        print("Initialize training, validation, and test indices")

        self.for_faceformer = for_faceformer
        if self.for_faceformer:
            self._init_indices_v2()
        else:
            self._init_indices()

    def get_data_splits(self):
        return self.training_indices, self.validation_indices, self.testing_indices

    def slice_data(self, indices):
        if self.for_faceformer:
            return self._slice_data_helper_v2(indices)
        return self._slice_data(indices)

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

    def _init_indices(self):
        def get_indices(subjects, sequences):
            indices = []
            for subj in subjects:
                if (subj not in self.raw_audio) or (subj not in self.data2array_verts):
                    if subj != '':
                        import pdb; pdb.set_trace()
                        print('subject missing %s' % subj)
                    continue

                for seq in sequences:
                    if (seq not in self.raw_audio[subj]) or (seq not in self.data2array_verts[subj]):
                        print('sequence data missing %s - %s' % (subj, seq))
                        continue

                    num_data_frames = max(self.data2array_verts[subj][seq].keys())+1
                    if self.processed_audio is not None:
                        num_audio_frames = len(self.processed_audio[subj][seq]['audio'])
                    else:
                        num_audio_frames = num_data_frames

                    try:
                        for i in range(min(num_data_frames, num_audio_frames)):
                            indexed_frame = self.data2array_verts[subj][seq][i]
                            indices.append(indexed_frame)
                    except KeyError:
                        print('Key error with subject: %s and sequence: %s" % (subj, seq)')
            return indices

        self.training_indices = get_indices(self.training_subjects, self.training_sequences)
        self.validation_indices = get_indices(self.validation_subjects, self.validation_sequences)
        self.testing_indices = get_indices(self.testing_subjects, self.testing_sequences)

        self.training_idx2subj = {idx: self.training_subjects[idx] for idx in np.arange(len(self.training_subjects))}
        self.training_subj2idx = {self.training_idx2subj[idx]: idx for idx in self.training_idx2subj.keys()}

    def _init_indices_v2(self):
        """Initialize the indices for FaceFormer training
        """
        self.training_indices = self.all_training_sequences
        self.validation_indices = self.all_validation_sequences
        self.testing_indices = self.all_testing_sequences

        self.training_idx2subj = {idx: self.training_subjects[idx] for idx in np.arange(len(self.training_subjects))}
        self.training_subj2idx = {self.training_idx2subj[idx]: idx for idx in self.training_idx2subj.keys()}

    def _slice_data(self, indices):
        if self.num_consecutive_frames == 1:
            return self._slice_data_helper(indices)
        else:
            window_indices = []
            for id in indices:
                window_indices += self.array2window_ids[id]
            return self._slice_data_helper(window_indices)

    def _slice_data_helper(self, indices):
        face_vertices = self.face_vert_mmap[indices]
        face_templates = []
        processed_audio = []
        subject_idx = []
        for idx in indices:
            sub, sen, frame = self.array2data_verts[idx]
            face_templates.append(self.templates_data[sub])
            if self.processed_audio is not None:
                processed_audio.append(self.processed_audio[sub][sen]['audio'][frame])
            subject_idx.append(self.convert_training_subj2idx(sub))

        face_templates = np.stack(face_templates)
        subject_idx = np.hstack(subject_idx)
        assert face_vertices.shape[0] == face_templates.shape[0]

        if self.processed_audio is not None:
            processed_audio = np.stack(processed_audio)
            assert face_vertices.shape[0] == processed_audio.shape[0]
        return processed_audio, face_vertices, face_templates, subject_idx

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
        batch_data_dict['face_vertices'] = np.stack(batched_face_vertices)
        batch_data_dict['face_template'] = np.stack(batched_face_template)
        batch_data_dict['subject_idx'] = np.stack(batched_subject_idx)
        try:
            batch_data_dict['raw_audio'] = np.stack(batched_raw_audio)
        except:
            all_shape = [s.shape for s in batched_raw_audio]
            print(all_shape)
 
        return batch_data_dict

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

    def _load_data(self, config):
        face_verts_mmaps_path = load_from_config(config, 'verts_mmaps_path')
        face_templates_path = load_from_config(config, 'templates_path')
        raw_audio_path = load_from_config(config, 'raw_audio_path')
        processed_audio_path = load_from_config(config, 'processed_audio_path')
        data2array_verts_path = load_from_config(config, 'data2array_verts_path')

        print("Loading face vertices")
        self.face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r')

        print("Loading templates")
        self.templates_data = pickle.load(open(face_templates_path, 'rb'), encoding='latin1')

        print("Loading raw audio")
        self.raw_audio = pickle.load(open(raw_audio_path, 'rb'), encoding='latin1')

        print("Process audio")
        self.processed_audio = pickle.load(open(processed_audio_path, 'rb'), encoding='latin1')

        print("Loading index maps")
        self.data2array_verts = pickle.load(open(data2array_verts_path, 'rb'))
        self.array2data_verts = invert_data2array(self.data2array_verts)
        self.array2window_ids = compute_window_array_idx(self.data2array_verts, self.num_consecutive_frames)

    def _init_data_splits(self, subject_for_training, sequence_for_training, subject_for_validation,
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
            raw_audio.append(self.raw_audio[subj][seq])
            processed_seq_audio = []
            if self.processed_audio is not None:
                for frame, array_idx in self.data2array_verts[subj][seq].items():
                    processed_seq_audio.append(self.processed_audio[subj][seq]['audio'][frame])
            processed_audio.append(processed_seq_audio)
        return raw_audio, processed_audio, face_vertices, face_templates, subject_idx
