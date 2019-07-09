import numpy as np


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img


def window_normalize(img, WW, WL, dst_range=(0, 1)):
    """
    WW: window width
    WL: window level
    dst_range: normalization range
    """
    src_min = WL - WW/2
    src_max = WL + WW/2
    outputs = (img - src_min)/WW * (dst_range[1] - dst_range[0]) + dst_range[0]
    outputs[img >= src_max] = 1
    outputs[img <= src_min] = 0
    return outputs

