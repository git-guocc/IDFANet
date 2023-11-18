import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class ShapeNetDataset(Dataset):
    def __init__(self, data_path, split, transform=None):
        assert split in {"train", "val", "test"}
        self.mode = split
        self.data_path = data_path
        self.transform = transform
        self.data_patial = self._load_data_patial()
        self.data_gt = self._load_data_gt()

    def __len__(self):
        return len(self.data_gt)

    def _load_data_patial(self):
        split_file = os.path.join(self.data_path, f'{self.mode}.list')
        with open(split_file, 'r') as file:
            data_list = file.read().splitlines()
        return data_list

    def _load_data_gt(self):
        split_file = os.path.join(self.data_path, f'{self.mode}.list')
        with open(split_file, 'r') as file:
            data_list = file.read().splitlines()
        return data_list

    def __getitem__(self, index):
        # 加载部分点云
        with h5py.File(self.data_path + f"{self.mode}/partial/{self.data_patial[index]}.h5", 'r') as f:
            in_xyz = f['data'][:].astype('float32')  # 转换为 float32 类型
            in_xyz = torch.from_numpy(in_xyz)  # 转换为 torch Tensor

        # 加载完整点云
        with h5py.File(self.data_path + f"{self.mode}/gt/{self.data_gt[index]}.h5", 'r') as f:
            gt_xyz = f['data'][:].astype('float32')  # 转换为 float32 类型
            gt_xyz = torch.from_numpy(gt_xyz)  # 转换为 torch Tensor

        return in_xyz, gt_xyz


if __name__ == '__main__':
    x = ShapeNetDataset(data_path='/home/guocc/GitHub/PointCloud/SceneCompletion/IDFANet/data/shapenet/', split='val')
    sample = x[0]  # 获取第一个样本
    in_xyz, gt_xyz = sample
    print("Input XYZ shape:", in_xyz.shape)
    print("Ground Truth XYZ shape:", gt_xyz.shape)
