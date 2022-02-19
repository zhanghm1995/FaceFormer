'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-18 19:26:02
Email: haimingzhang@link.cuhk.edu.cn
Description: The batcher class to load VOCASET dataset for training
'''

import copy
import random
import numpy as np
from torch.utils.data.dataset import Dataset


class Batcher(Dataset):
    def __init__(self, data_handler):
        self.data_handler = data_handler

        data_splits = data_handler.get_data_splits()
        self.training_indices = data_splits[0]
        self.val_indices = data_splits[1]
        self.test_indices = data_splits[2]

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
            batch_indices = self.training_indices[self.current_state: (self.current_state + batch_size)]
            self.current_state += batch_size
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