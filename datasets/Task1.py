import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
from utils.preprocessing import window_normalize

class dataset(Dataset):
    n_classes = 22
    void_classes = [0]
    valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    class_map = dict(zip(valid_classes, range(1, n_classes+1)))

    def __init__(self, root, ignore_index=0):
        self.ignore_index = ignore_index
        self.samples = [os.path.join(root, img_name) for img_name in os.listdir(os.path.join(root))]
        self.outputs_channels = self.n_classes+1
        # add transform

    def __getitem__(self, index):
        sample_path = self.samples[index]
        volume_path = os.path.join(sample_path, 'data.nii.gz')
        gt_path = os.path.join(sample_path, 'label.nii.gz')

        itk_CT = sitk.ReadImage(volume_path)
        itk_gt = sitk.ReadImage(gt_path)
        torch_CT = self._img_transfor(itk_CT)
        torch_gt = self._label_transfor(itk_gt)
        return torch_CT, torch_gt

    def __len__(self):
        return len(self.samples)

    def _img_transfor(self, itk):
        img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
        img_arr = window_normalize(img_arr, WW=350, WL=40)
        torch_itk = torch.from_numpy(img_arr)
        return torch_itk

    def _label_transfor(self, itk):
        img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
        # encode mask
        for c in self.void_classes:
            img_arr[img_arr == c] = self.ignore_index
        for c in self.valid_classes:
            img_arr[img_arr == c] = self.class_map[c]
        torch_itk = torch.from_numpy(img_arr.astype(np.long))
        return torch_itk