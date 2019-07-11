import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
from utils.preprocessing import window_normalize

class dataset(Dataset):
    n_classes = 1

    def __init__(self, root, ignore_index=0):
        self.ignore_index = ignore_index
        self.samples = [os.path.join(root, img_name) for img_name in os.listdir(os.path.join(root))]
        self.outputs_channels = self.n_classes
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
        img_arr = window_normalize(img_arr, WW=1500, WL=-500)
        torch_itk = torch.from_numpy(img_arr)
        return torch_itk

    def _label_transfor(self, itk):
        img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
        torch_itk = torch.from_numpy(img_arr)
        return torch_itk