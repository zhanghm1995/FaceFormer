'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-18 20:19:02
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import random
from .voca_dataset import DataHandler
from .voca_dataset_batcher import Batcher
from torch.utils.data import DataLoader

import torch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

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
    

def get_random_fixed_2d_dataset(config, split, num_sequences):
    from .face_image_dataset import FaceImageDataset

    dataset = FaceImageDataset(data_root=config['data_root'], split=split)
    seq_list = list(range(len(dataset)))
    
    st = random.getstate()
    random.seed(777)
    random.shuffle(seq_list)
    random.setstate(st)

    if num_sequences > 0 and num_sequences < len(dataset):
        seq_list = seq_list[:num_sequences]
    
    data_list = []
    for idx in seq_list:
        data_list.append(dataset[idx])
    return data_list


def get_2d_3d_dataset(config, split):
    from .face_2d_3d_dataset import Face2D3DDataset

    dataset = Face2D3DDataset(data_root=config['data_root'], split=split)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(split=="train"),
        num_workers=config['number_workers'],
        # pin_memory=True,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return data_loader


def get_random_fixed_2d_3d_dataset(config, split, num_sequences):
    from .face_2d_3d_dataset import Face2D3DDataset

    dataset = Face2D3DDataset(data_root=config['data_root'], split=split, fetch_length=200)
    seq_list = list(range(len(dataset)))
    
    st = random.getstate()
    random.seed(111)
    random.shuffle(seq_list)
    random.setstate(st)

    if num_sequences > 0 and num_sequences < len(dataset):
        seq_list = seq_list[:num_sequences]
    
    print("Selected indices: ", seq_list)
    sub_dataset = []
    for idx in seq_list:
        sub_dataset.append(dataset[idx])

    data_loader = DataLoader(
        sub_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['number_workers'],
        # pin_memory=True,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return data_loader

    return data_list