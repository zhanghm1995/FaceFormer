'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 19:29:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the 2D 3D Dataset
'''

from omegaconf import OmegaConf
from dataset import get_dataset
from tqdm import tqdm


def test_2d_3d_dataset():
    config = OmegaConf.load("./config/face_3dmm_motion_mouth_mask_pix2pixhd.yaml")

    dataset_config = config['dataset']
    dataset_config['dataset_name'] = "Face2D3DDataset"

    train_dataloader = get_dataset(config['dataset'], split="voca_train", shuffle=True)
    print(len(train_dataloader))

    dataset = next(iter(train_dataloader))

    print(dataset["video_name"])

    for i, dataset in tqdm(enumerate(train_dataloader)):
        print(dataset["video_name"])


if __name__ == "__main__":
    test_2d_3d_dataset()