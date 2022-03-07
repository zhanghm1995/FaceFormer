'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-25 09:41:03
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the Face2D3DDataset class
'''

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

if __name__ == "__main__":
    pass



    