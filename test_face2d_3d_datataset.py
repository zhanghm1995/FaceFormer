'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-25 09:41:03
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the Face2D3DDataset class
'''

from dataset.face_2d_3d_dataset import Face2D3DDataset

if __name__ == "__main__":
    data_root = "/home/haimingzhang/Research/Face/FaceFormer/FaceFormer/data/id00002"
    split = "train"
    dataset = Face2D3DDataset(data_root, split)
    print(len(dataset))

    data = dataset[180]
    print(data.shape)
