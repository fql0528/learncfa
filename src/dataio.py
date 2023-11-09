from src.utils import *
from torch.utils.data import Dataset
import torch
import numpy as np


def torch_rgb2gray(vid):
    # weights from Wikipedia 从维基百科的权重
    vid[:, 0, :, :] *= 0.2126
    vid[:, 1, :, :] *= 0.7152
    vid[:, 2, :, :] *= 0.0722
    return torch.sum(vid, dim=1)


def crop_center(img, size_x, size_y):
    _, _, y, x = img.shape
    # print('x',x)
    # print('y',y)
    startx = x // 2 - (size_x // 2)
    starty = y // 2 - (size_y // 2)
    # print('startx',startx)
    # print('starty',starty)
    return img[..., starty:starty + size_y, startx:startx + size_x]


class NFS_Video(Dataset):
    def __init__(self,
                 log_root='../custom_data/nfs_block_rgb_512_8f',
                 block_size=[8, 512, 512],
                 gt_index=0,
                 split='train',
                 color=False,
                 test=False):
        '''
        3 x 1280 x 720 pixels originally 原来是3 x 1280 x 720像素
        init blocks will make it 512 x 512 Init块将使其为512 x 512
        '''
        super().__init__()

        self.log_root = log_root
        self.block_size = block_size
        self.split = split
        self.gt_index = gt_index
        self.color = color
        self.test = test
        # load video block names load video block names
        print('creating list of video blocks创建视频块列表') #原版
        self.video_blocks = []
        print('self.log_root',self.log_root) #原版

        # fpath = f'{self.log_root}/{self.split}/blocks_per_vid8.pt' #原版
        fpath = f'{self.log_root}/{self.split}/nfs_block.pt'
        if self.split == 'sample':
            raise NotImplementedError('no sample dataset for nfs')

        self.vid_dict = torch.load(fpath)

        # print('len(self.vid_dict[0])',len(self.vid_dict[0])) # 80
        # print('len(self.vid_dict[1])',len(self.vid_dict[1])) #102
        # print('self.vid_dict',self.vid_dict) #72个tensor，为字典从tensor0--tensor71
        # print('len(self.vid_dict)',len(self.vid_dict)) #1
        # print('self.vid_dict.keys()',self.vid_dict.keys()) #dict_keys([0, 1])

        # dict = {key for key,value in self.vid_dict.items() } 
        # print(dict) #{0}

        self.num_vids = max(self.vid_dict.keys()) + 1  #有几个视频

        self.num_clips_each = [] #剪辑每个视频
        for i in range(self.num_vids):
            vid = self.vid_dict[i]
            # print('vid',vid)  #第i个视频，这里就是第一个视频只有一个视频和self.vid_dict相同
            self.num_clips_each.append(max(vid.keys()) + 1)
            # print(self.num_clips_each) #[80, 102]

        self.num_clips_total = sum(self.num_clips_each) #182
        # self.num_clips_total = self.num_vids * self.num_clips_each
        print(f'loaded {self.num_clips_total} clips from {self.num_vids} videos')

        self.stop_idx = np.cumsum(np.array(self.num_clips_each))
        # print('self.stop_idx',self.stop_idx) # [80 182]
        """
        numpy.cumsum(a, axis=None, dtype=None, out=None)
        axis=0，按照行累加。
        axis=1，按照列累加。
        axis不给定具体值，就把numpy数组当成一个一维数组。
        """
        vid_mapping = {}

        vid_num = 0
        clip_num = 0

        # convert integer index to right video corresponding to 转换整数索引到对应的右视频
        # number of videos and clips of each video 视频的数量和每个视频的剪辑
        for i in range(self.num_clips_total): #self.num_clips_total  182
            if i == self.stop_idx[vid_num]: #self.stop_idx  [80 182]
                vid_num += 1
                clip_num = 0
            vid_mapping[i] = (vid_num, clip_num)
            clip_num += 1
        self.vid_mapping = vid_mapping
        # print('self.vid_mapping',self.vid_mapping) #{0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6), 

    def __len__(self):
        return self.num_clips_total

    def __getitem__(self, idx):
        # print('idx',idx)
        (vid_num, clip_num) = self.vid_mapping[idx]
        # print('vid_num',vid_num)# 0
        # print('clip_num',clip_num)#
        vid = self.vid_dict[vid_num][clip_num]  # 8, 3, H, W
        # print('vid.shape',vid.shape) # torch.Size([8, 3, 512, 512])

        if self.block_size[-1] != 512:
       
            vid = crop_center(vid, self.block_size[-2], self.block_size[-1])

        if self.color:
            gt = vid[self.gt_index, ...]
            avg = torch.mean(vid, dim=0)
        else:
            # print('vid.shape',vid.shape) #torch.Size([8, 3, 512, 512]) 
            vid = torch_rgb2gray(vid.clone()) # 将真彩色图像 RGB 转换为灰度图像
            # print('vid',vid)
            avg = torch.mean(vid, dim=0, keepdim=True) # [1, H, W]
            # print('avg',avg)
            # print('self.gt_index',self.gt_index) #0
            # print('vid.shape',vid.shape) #torch.Size([8, 512, 512])
            gt = vid[self.gt_index, ...] #torch.Size([512, 512])
            # print('gt',gt) #torch.Size([512, 512])
            # print('gt.shape',gt.shape)
            gt = gt.unsqueeze(0) # [1, H, W]
            # print('gt1',gt) #torch.Size([1,512, 512])
            if not self.test:
                [avg, vid, gt] = augmentData([avg, vid, gt])
            # avg [1,h,w]
            # vid [8,h,w]
            # gt  [1,h,w]
        return avg, vid, gt
