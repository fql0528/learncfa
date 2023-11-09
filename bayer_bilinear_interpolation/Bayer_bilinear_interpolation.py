import torch
from torch.nn import functional as F

def initialize(input,device):
    r""" Initialize with bilinear interpolation
    输入为4维，batch_size channels height weight
    """
    F_r = torch.FloatTensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    F_b = F_r
    F_g = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    # print('F_r',F_r)
    # print('F_g',F_g)
    bilinear_filter = torch.stack([F_r, F_g, F_b])[:, None]
    # print(bilinear_filter)
    # print('bilinear_filter.shape',bilinear_filter.shape) #torch.Size([3, 1, 3, 3])
    bilinear_filter = bilinear_filter.to(device)
    res = F.conv2d(input, bilinear_filter, padding=1, groups=3)
    return res