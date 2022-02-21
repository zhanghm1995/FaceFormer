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

def get_2d_dataset(config, split):
    from .face_image_dataset import FaceImageDataset
    from torch.utils.data import DataLoader

    dataset = FaceImageDataset(data_root=config['data_root'], split=split)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(split=="train"),
        num_workers=config['number_workers'],
        # pin_memory=True,
        pin_memory=False,
    )
    return data_loader
    