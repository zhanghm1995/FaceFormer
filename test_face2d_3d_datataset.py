'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-25 09:41:03
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the Face2D3DDataset class
'''

import torch
from dataset.face_2d_3d_dataset import Face2D3DDataset
from dataset.face_2d_3d_test_dataset import Face2D3DTestDataset
from utils.utils import tensor2im
from utils.save_data import save_image_array_to_video


def test_Face2D3DDataset():
    data_root = "/home/haimingzhang/Research/Face/FaceFormer/FaceFormer/data/id00002"
    split = "train"
    dataset = Face2D3DDataset(data_root, split, load_mouth_mask=False, load_ref_image=False)
    print(len(dataset))

    data = dataset[180]

    for key, value in data.items():
        print(key, value.shape)
    
    input_image = data['input_image']
    gt_face_image = data['gt_face_image']

    save_image_array_to_video(input_image[None], "./temp", name="input")
    save_image_array_to_video(gt_face_image[None], "./temp", name="gt")

def test_Face2D3DTestDataset():
    data_root = "/home/haimingzhang/Research/Face/FaceFormer/FaceFormer/data/id00002"
    dataset = Face2D3DTestDataset(data_root, load_mouth_mask=False, load_ref_image=False)
    print(len(dataset))

    data = dataset[180]

    for key, value in data.items():
        print(key, value.shape)
    
    input_image = data['input_image']
    gt_face_image = data['gt_face_image']

    save_image_array_to_video(input_image[None], "./temp", name="input")

    save_image_array_to_video(gt_face_image[None], "./temp", name="gt")

def test_Face2D3DDataLoader():
    from dataset import get_2d_3d_dataset
    from omegaconf import OmegaConf
    import torchvision
    import numpy as np

    config = OmegaConf.load("config/config_2d_3d_fusion_HDTF.yaml")

    train_dataloader = get_2d_3d_dataset(config['dataset'], split="train", shuffle=True)

    data = next(iter(train_dataloader))
    print(type(data))
    
    for key, value in data.items():
        print(key)
        try:
            print(value.shape)
        except:
            pass
    
    input_image = data['gt_face_image'][0]

    numpy_image = input_image.numpy()
    input_image = np.ascontiguousarray(numpy_image[:, ::-1, ...])
    print(input_image.shape)
    input_image = torch.from_numpy(input_image)
    grid_image = torchvision.utils.make_grid(input_image, nrow=25)
    torchvision.utils.save_image(grid_image, fp=f'gt_face_image4.png')

    input_image = data['input_image'][0]

    numpy_image = input_image.numpy()
    input_image = np.ascontiguousarray(numpy_image[:, ::-1, ...])
    print(input_image.shape)
    input_image = torch.from_numpy(input_image)
    grid_image = torchvision.utils.make_grid(input_image, nrow=25)
    torchvision.utils.save_image(grid_image, fp=f'input_image4.png')


if __name__ == "__main__":
    test_Face2D3DDataLoader()



    