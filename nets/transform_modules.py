import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pykdtree.kdtree import KDTree #有问题
from scipy.spatial import KDTree


# device = 'cuda:0'
# from scipy.spatial import KDTree

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def norm_coord(coords, length=512):
    ''' Convert image from [0, 512] pixel length to [-1, 1] coords 将图像从[0,512]像素长度转换为[- 1,1]坐标'''
    coords = coords / length
    coords -= 0.5
    coords *= 2
    return coords.detach().cpu().numpy()


class TreeMultiRandom(nn.Module):
    def __init__(self, sz=[512,512], k=3, p=1, num_channels=8):
        super().__init__()

        self.k = k # number of neighboring points to search 要搜索的相邻点的个数
        self.p = p
        self.sz = sz
        self.num_channels = num_channels

    def _find_idx(self, mask):
        # print('mask',mask)
        # print('mask.shape',mask.shape) #torch.Size([1,512, 512]) 里面全是bool值
        # img = torch.ones((self.sz, self.sz), device=device) #原版
        # img = torch.ones((self.sz[0], self.sz[1]), device=device) #非原版
        img=torch.ones((mask.shape[1],mask.shape[2]),device=device) #非原版
        # print('img.shape',img.shape) #torch.Size([1359, 2041])
        holed_img = img * mask
        # print('holed_img',holed_img)
        # filled_idx = (holed_img != 0).nonzero()  # 原版filled_idx ：填充索引
        # unfilled_idx = (holed_img == 0).nonzero() #原版
        #nonzero()获取所有非零元素的下标
        filled_idx = (holed_img != 0).nonzero(as_tuple=False)
        # print('filled_idx',filled_idx) #返回的不为0的元素的索引
        unfilled_idx = (holed_img == 0).nonzero(as_tuple=False)
        # print('unfilled_idx',unfilled_idx) #返回的为0的元素的索引

        # filled_idx_n = norm_coord(filled_idx, self.sz)  # 原版 num_coords, 2 将图像从[0,512]像素长度转换为[- 1,1]坐标
        # filled_idx_n = norm_coord(filled_idx, self.sz[0])  #非原版
        filled_idx_n = norm_coord(filled_idx, mask.shape[1])  # 非原版
        # print('filled_idx_n',filled_idx_n)
        # unfilled_idx_n = norm_coord(unfilled_idx, self.sz)  #原版 num_coords, 2
        # unfilled_idx_n = norm_coord(unfilled_idx, self.sz[0]) #非原版
        unfilled_idx_n = norm_coord(unfilled_idx, mask.shape[1])  # 非原版
        # print('unfilled_idx_n',unfilled_idx_n)

        tree = KDTree(filled_idx_n)
        # print('tree',tree) #<pykdtree.kdtree.KDTree object at 0x0000021B893BF358>
        # print('tree.data',tree.data) #tree.data [-1.         -0.99609375 -0.99609375 ... -1.          0.99609375 0.99609375]
        # print('tree.data.shape',tree.data.shape) # (196608,)

        #https://www.cnblogs.com/yibeimingyue/p/13797529.html
        dist, idx = tree.query(unfilled_idx_n, k=self.k) #返回值是：离查询点最近的点的距离和索引

        idx = idx.astype(np.int32)
        dist = torch.from_numpy(dist).to(device) #将numpy数组转换为PyTorch中的张量
        # print('idx',idx)#
        # print('idx.shape',idx.shape)#(196608, 3)  (131072, 3)
        # print('dist',dist)
        # print('dist.shape',dist.shape)#torch.Size([196608, 3])  torch.Size([131072, 3])
        return idx, dist, filled_idx, unfilled_idx

    def _fill(self, holed_img, params):
        # print('holed_img.shape',holed_img.shape) #torch.Size([1, 512, 512])
        b, _, _ = holed_img.shape #torch.Size([2, 512, 512])

        idx, dist, filled_idx, unfilled_idx = params
        # print('idx.shape',idx.shape)
        # print('filled_idx',filled_idx)
        # idx = num_coords, k
        # filled_idx = num_coords, 2
        
        vals = torch.zeros((b, dist.shape[0]), dtype=torch.float32, device=device)


        for i in range(self.k):
            # find coords of the points which are knn  求出KNN点的坐标

            idx_select = filled_idx[idx[:, i]]  # num_coords, k

            # add value of those coords, weighted by their inverse distance 加上这些坐标的值，用它们的反距离加权
            # print('holed_img',holed_img)
            # holed_img.register_hook(print)  # 不为
            vals += holed_img[:, idx_select[:, 0], idx_select[:, 1]] * (1.0 / dist[:, i]) ** self.p
            # vals.register_hook(print)  # 不为0
        vals /= torch.sum((1.0 / dist) ** self.p, dim=1)
        # print('(1.0 / dist).shape',(1.0 / dist).shape)
        # vals.register_hook(print) #不为0

        holed_img[:, unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        # holed_img.register_hook(print)
        # print('holed_img1111111111111',holed_img)
        return holed_img

    def shutter_length(self,coded):
        self.length = torch.ones((coded.shape[2],coded.shape[3]), dtype=torch.float32, device=device)
        self.length[::4, 2::4] = 1
        self.length[1::4, 3::4] = 1
        self.length[::4, ::4] = 2
        self.length[1::4, 1::4] = 2
        self.length[2::4, 2::4] = 2
        self.length[3::4, 3::4] = 2
        self.length[2::4, ::4] = 3
        self.length[3::4, 1::4] = 3
        self.length[::2, 1::2] = 4
        self.length[1::2, ::2] = 4
        # print('self.length',self.length)
        # print('self.length.shape',self.length.shape) #torch.Size([512, 512])
        self.length = self.length.unsqueeze(0)  # torch.Size([1,512, 512])
        return  self.length
    def forward(self, coded, shutter_len):
        # print('coded.shape',coded.shape)
        b, _, h, w = coded.shape #torch.Size([2, 1, 512, 512])

        stacked = torch.zeros((b, self.num_channels, h, w), device=device)  # torch.Size([2, 4, 512, 512])
        # print('stacked.shape',stacked.shape) #torch.Size([1, 9, 512, 512])
        # print('shutter_len',shutter_len)

        """
        Quad
        tensor([[[8., 4., 8.,  ..., 4., 8., 4.],
         [4., 1., 4.,  ..., 1., 4., 1.],
         [8., 4., 8.,  ..., 4., 8., 4.],
         ...,
         [4., 1., 4.,  ..., 1., 4., 1.],
         [8., 4., 8.,  ..., 4., 8., 4.],
         [4., 1., 4.,  ..., 1., 4., 1.]]], device='cuda:0')
        """
        # shutter_len=shutter_len.unsqueeze(0) #  vscode原版
        if self.num_channels == 9 or self.num_channels == 4: # learn_all
            # print(1111111)
            # print('shutter_len',shutter_len)
            # print('shutter_len.shape',shutter_len.shape)
            # unique_lengths = torch.unique(shutter_len).type(torch.int8) #原版
            unique_lengths = torch.unique(shutter_len.detach()).type(torch.int8)
            # print('unique_lengths',torch.unique(shutter_len.detach()))
            # print('unique_lengths',unique_lengths) #tensor([1, 2, 4, 5, 8], device='cuda:0', dtype=torch.int8)
            # print('unique_lengths',unique_lengths) #  Quad tensor([1, 4, 8], device='cuda:0', dtype=torch.int8)           
            for i, length in enumerate(unique_lengths):
                # print('i',i) #0 1 2 3 4    Quad  0 1 2
                # print('length',length) #1 2 4 5 8   Quad  tensor(1, device='cuda:0', dtype=torch.int8) 4, 8
                # shutter_len=self.shutter_length(coded) #非原版

                mask = (shutter_len == length)  #原版      # 512, 512
                # print('mask',mask) #里面全是bool值
                # print('mask.shape',mask.shape) #torch.Size([1,512, 512])
                # print('coded',coded)
                # print('coded.shape',coded.shape) #torch.Size([2, 1, 512, 512])
                # print('coded[:, 0, :, :]', coded[:, 0, :, :])
                # print('coded[:, 0, :, :].shape', coded[:, 0, :, :].shape) # torch.Size([2, 512, 512])
                holed_img = coded[:, 0, :, :] * mask    # 4, 512, 512 (remove empty axis  移除空轴)
                # print('holed_img.shape',holed_img.shape) #torch.Size([1, 512, 512])
                # print('holed_img',holed_img)
                params = self._find_idx(mask)  #idx, dist, filled_idx, unfilled_idx
                # print('params',params)
                # print('holed_img.shape',holed_img.shape) #torch.Size([1, 512, 512])
                # print(holed_img==0)
                filled_img = self._fill(holed_img, params)
                # print('filled_img',filled_img)
                # print('filled_img.shape',filled_img.shape) #torch.Size([2, 512, 512])
                stacked[:, i, :, :] = filled_img
                # print('i',i) #0 1 2
                # print('stacked.shape',stacked.shape) #torch.Size([2, 4, 512, 512])
        else:
            raise NotImplementedError('this has not been implemented')
        return stacked


class TileInterp(nn.Module):
    def __init__(self, shutter_name, tile_size, sz, interp='bilinear'):
        super().__init__()
        self.shutter_name = shutter_name
        self.tile_size = tile_size
        self.sz = sz
        self.interp = interp

    def forward(self, coded):
        b, _, _, _ = coded.shape #torch.Size([2, 1, 512, 512])
        full_stack = torch.zeros((b, self.tile_size ** 2, self.sz, self.sz), dtype=torch.float32, device=device)
        # print('full_stack.shape',full_stack.shape) #torch.Size([2, 4, 512, 512])
        curr_channel = 0
        for i in range(self.tile_size):
            for j in range(self.tile_size):
                # 1,1,H/3,W/3 getting every measurement in the tile
                sampled_meas = coded[:, :, i::self.tile_size, j::self.tile_size]
                # print('sampled_meas.shape',sampled_meas.shape)# torch.Size([2, 1, 256, 256]) 
                full_res = F.interpolate(sampled_meas, size=[self.sz, self.sz], mode='bilinear', align_corners=True)
                full_stack[:, curr_channel, ...] = full_res.squeeze(1)
                curr_channel += 1
        # print('full_stack.shape',full_stack.shape) #torch.Size([2, 4, 512, 512])
        return full_stack
