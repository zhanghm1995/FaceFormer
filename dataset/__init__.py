'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-18 20:19:02
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

from .voca_dataset import DataHandler
from .voca_dataset_batcher import Batcher


def get_dataset(config):
    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)
    return batcher
