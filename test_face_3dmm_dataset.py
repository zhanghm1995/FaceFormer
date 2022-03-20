'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 20:53:40
Email: haimingzhang@link.cuhk.edu.cn
Description: Test Face3DMMDataset
'''

from easydict import EasyDict
from dataset import get_3dmm_dataset


def test_face_3dmm_dataset():
    config = EasyDict(data_root="./data/HDTF_preprocessed", fetch_length=75, batch_size=2, number_workers=4)

    train_dataloader = get_3dmm_dataset(config, split="train")
    print(len(train_dataloader))

    dataset = next(iter(train_dataloader))

    for key, value in dataset.items():
        print(key)
        print(value.shape)


if __name__ == "__main__":
    test_face_3dmm_dataset()