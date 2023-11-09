import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykdtree.kdtree import KDTree
import scipy.spatial as kd
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

class MyTreeMultiRandom(nn.Module):
    def __init__(self, k=3, p=1, num_channels=8):
        super().__init__()
        self.k = k  # number of neighboring points to search 要搜索的相邻点的个数
        self.p = p
        # self.sz = sz
        self.num_channels = num_channels

    def _find_idx(self, mask):
        # print('mask.shape',mask.shape) #torch.Size([1,512, 512]) 里面全是bool值
        # img = torch.ones((self.sz, self.sz), device=device) #原版
        # img = torch.ones((self.sz[0], self.sz[1]), device=device) #非原版
        img = torch.ones((mask.shape[1], mask.shape[2]), device=device)  # 非原版
        # print('img.shape',img.shape) #torch.Size([1359, 2041])
        holed_img = img * mask
        # print(holed_img==mask)
        # print('holed_img',holed_img)
        # filled_idx = (holed_img != 0).nonzero()  # 原版filled_idx ：填充索引
        # unfilled_idx = (holed_img == 0).nonzero() #原版
        # nonzero()获取所有非零元素的下标
        filled_idx = (holed_img != 0).nonzero(as_tuple=False)
        # print('filled_idx',filled_idx) #返回的不为0的元素的索引
        unfilled_idx = (holed_img == 0).nonzero(as_tuple=False)
        # print('unfilled_idx',unfilled_idx) #返回的为0的元素的索引

        # filled_idx_n = norm_coord(filled_idx, self.sz)  # 原版 num_coords, 2 将图像从[0,512]像素长度转换为[- 1,1]坐标
        # filled_idx_n = norm_coord(filled_idx, self.sz[0])  #非原版
        filled_idx_n = norm_coord(filled_idx, mask.shape[2])  # 非原版
        # print('filled_idx_n',filled_idx_n)
        # unfilled_idx_n = norm_coord(unfilled_idx, self.sz)  #原版 num_coords, 2
        # unfilled_idx_n = norm_coord(unfilled_idx, self.sz[0]) #非原版
        unfilled_idx_n = norm_coord(unfilled_idx, mask.shape[2])  # 非原版
        # print('unfilled_idx_n',unfilled_idx_n)

        tree = KDTree(filled_idx_n)
        # tree=kd(filled_idx_n)
        # print('tree',tree) #<pykdtree.kdtree.KDTree object at 0x0000021B893BF358>
        # print('tree.data',tree.data) #tree.data [-1.         -0.99609375 -0.99609375 ... -1.          0.99609375 0.99609375]
        # print('tree.data.shape',tree.data.shape) # (196608,)

        # https://www.cnblogs.com/yibeimingyue/p/13797529.html
        dist, idx = tree.query(unfilled_idx_n, k=self.k)  # 返回值是：离查询点最近的点的距离和索引

        idx = idx.astype(np.int32)
        dist = torch.from_numpy(dist).to(device)  # 将numpy数组转换为PyTorch中的张量
        # print('idx',idx)#
        # print('idx.shape',idx.shape)#(196608, 3)  (131072, 3)
        # print('dist',dist)
        # print('dist.shape',dist.shape)#torch.Size([196608, 3])  torch.Size([131072, 3])
        return idx, dist, filled_idx, unfilled_idx

    def _fill(self, holed_img, params):
        b, _, _ = holed_img.shape  # torch.Size([2, 512, 512])

        idx, dist, filled_idx, unfilled_idx = params
        # idx = num_coords, k
        # filled_idx = num_coords, 2
        vals = torch.zeros((1, dist.shape[0]), dtype=torch.float32, device=device)

        for i in range(self.k):
            # find coords of the points which are knn  求出KNN点的坐标
            idx_select = filled_idx[idx[:, i]]  # num_coords, k

            # add value of those coords, weighted by their inverse distance 加上这些坐标的值，用它们的反距离加权
            vals += holed_img[idx_select[:, 0], idx_select[:, 1]] * (1.0 / dist[:, i]) ** self.p
        vals /= torch.sum((1.0 / dist) ** self.p, dim=1)

        holed_img[unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        holed_img=holed_img.unsqueeze(0)
        return holed_img

    def shutter_length(self, coded):
        self.length = torch.ones((coded.shape[2], coded.shape[3]), dtype=torch.float32, device=device)
        if coded.shape[1]==3:
            self.length[1::2,::2] = 3.0
            self.length[::2, ::2] = 2.0
            self.length[1::2, 1::2] = 2.0
            self.length[::2, 1::2] = 1.0
        elif coded.shape[1]==4:
        # """
            self.length[::4, 2::4] = 1.0
            self.length[1::4, 3::4] = 1.0
            self.length[::4, ::4] = 2.0
            self.length[1::4, 1::4] = 2.0
            self.length[2::4, 2::4] = 2.0
            self.length[3::4, 3::4] = 2.0
            self.length[2::4, ::4] = 3.0
            self.length[3::4, 1::4] = 3.0
            self.length[::2, 1::2] = 4.0
            self.length[1::2, ::2] = 4.0
        # """
        else:
            print('shutter_length error')

        # print('self.length',self.length)
        # print('self.length.shape',self.length.shape) #torch.Size([512, 512])
        self.length = self.length.unsqueeze(0)  # torch.Size([1,512, 512])
        # print('self.length',self.length)
        return self.length

    # def forward(self, coded): #原来版本
    def forward(self, coded,shutter_len): #新版本
        # print('coded.shape',coded.shape) #torch.Size([1, 3, 186, 317])
        # shutter_len = self.shutter_length(coded) # 原来版本
        # print('shutter_len',shutter_len)
        # print('shutter_len.shape',shutter_len.shape) #torch.Size([512, 512])
        b, _, h, w = coded.shape  # torch.Size([2, 1, 512, 512])

        stacked = torch.zeros((b, self.num_channels, h, w), device=device)  # torch.Size([2, 4, 512, 512])
        # print('stacked.shape',stacked.shape) #torch.Size([2, 4, 512, 512])
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

        if self.num_channels == 3 or self.num_channels == 4:  # learn_all
            # print(1111111)
            # print('shutter_len',shutter_len)
            # unique_lengths = torch.unique(shutter_len).type(torch.int8) #原版
            unique_lengths = torch.unique(shutter_len.detach()).type(torch.int8) #原来版本

            # print('unique_lengths',unique_lengths) #tensor([1, 2, 3], device='cuda:0', dtype=torch.int8)
            # print('unique_lengths',unique_lengths) #  Quad tensor([1, 4, 8], device='cuda:0', dtype=torch.int8)
            # print('coded',coded)
            for i, length in enumerate(unique_lengths):
                # print('i',i) #0 1 2 3 4    Quad  0 1 2
                # print('length',length) #1 2 4 5 8   Quad  tensor(1, device='cuda:0', dtype=torch.int8) 4, 8
                # shutter_len = self.shutter_length(coded)  # 非原版
                # print('shutter_len',shutter_len)
                # print('length',length)
                # print('shutter_len.shape',shutter_len.shape)
                # print(length.shape,length.shape)
                mask = (shutter_len == length)  # 原版      # 512, 512
                # print('mask',mask) #里面全是bool值
                # print('mask.shape',mask.shape) #torch.Size([1,512, 512])
                # print('coded',coded)
                # print('coded.shape',coded.shape) #torch.Size([2, 1, 512, 512])
                # print('coded[:, 0, :, :]', coded[:, 0, :, :])
                # print('coded[:, 0, :, :].shape', coded[:, 0, :, :].shape) # torch.Size([2, 512, 512])
                holed_img = coded[:, i, :, :] * mask  # 4, 512, 512 (remove empty axis  移除空轴)
                # print('holed_img.shape',holed_img.shape) #torch.Size([2, 512, 512])
                # print('holed_img',holed_img)
                params = self._find_idx(mask)  # idx, dist, filled_idx, unfilled_idx
                filled_img = self._fill(holed_img, params)
                # print('filled_img',filled_img)
                # print('filled_img.shape',filled_img.shape) #torch.Size([2, 512, 512])
                stacked[:, i, :, :] = filled_img
                # print('i',i) #0 1 2
                # print('stacked.shape',stacked.shape) #torch.Size([2, 4, 512, 512])
        else:
            raise NotImplementedError('this has not been implemented')
        return stacked

class find_idx(nn.Module):
    def __init__(self,k=3):
        super(find_idx, self).__init__()
        self.k=k
    def forward(self,mask):
        img = torch.ones((mask.shape[0], mask.shape[1]), device=device)
        holed_img = img * mask
        unfilled_idx = (holed_img == 0).nonzero(as_tuple=False)
        filled_idx = (holed_img != 0).nonzero(as_tuple=False)
        filled_idx_n = filled_idx.detach().cpu().numpy()
        unfilled_idx_n = unfilled_idx.detach().cpu().numpy()
        tree = kd.KDTree(filled_idx_n)
        dist, idx = tree.query(unfilled_idx_n, k=self.k)
        idx = idx.astype(np.int32)
        dist = torch.from_numpy(dist).to(device)  # 将numpy数组转换为PyTorch中的张量
        return idx, dist, filled_idx_n, unfilled_idx_n
class fill(nn.Module):
    def __init__(self,k=3,p=1.0):
        super(fill, self).__init__()
        self.k=k
        self.p=p
    def forward(self,holed_img, params):
        b, _, _ = holed_img.shape
        idx, dist, filled_idx, unfilled_idx = params
        vals = torch.zeros((b, dist.shape[0]), dtype=torch.float32, device=device)
        # print('vals.shape',vals.shape)
        # vals = torch.zeros((1, dist.shape[0]), dtype=torch.float32,device=device)
        for i in range(self.k):
            idx_select = filled_idx[idx[:, i]]  # num_coords, k
            vals += holed_img[:, idx_select[:, 0], idx_select[:, 1]] * (1.0 / dist[:, i]) ** self.p
        vals /= torch.sum((1.0 / dist) ** self.p, dim=1)
        # print('holed_img.shape',holed_img.shape)
        holed_img[:, unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        print('holed_img[:,unfilled_idx[:, 0], unfilled_idx[:, 1]].shape',
              holed_img[:, unfilled_idx[:, 0], unfilled_idx[:, 1]].shape)
        # print('holed_img',holed_img)
        return holed_img


class MyTreeScatter(nn.Module):
    def __init__(self, k=3, p=1, num_channels=8):
        super().__init__()
        self.k = k  # number of neighboring points to search 要搜索的相邻点的个数
        self.p = p
        # self.sz = sz
        self.num_channels = num_channels
        self.find_idx=find_idx()
        self.fill=fill()
    def _find_idx(self,mask):
        # img = torch.ones((mask.shape[1], mask.shape[2]), device=device)
        # print('mask.shape',mask.shape) #torch.Size([4, 256, 256])
        # img = torch.ones((mask.shape[0], mask.shape[1]), device=device)
        # holed_img = img * mask
        unfilled_idx = (mask == 0).nonzero(as_tuple=False)
        filled_idx = (mask!= 0).nonzero(as_tuple=False)
        filled_idx_n=filled_idx.detach().cpu().numpy()
        unfilled_idx_n=unfilled_idx.detach().cpu().numpy()
        tree=kd.KDTree(filled_idx_n)
        dist, idx = tree.query(unfilled_idx_n, k=self.k)
        idx = idx.astype(np.int32)
        dist = torch.from_numpy(dist).to(device)  # 将numpy数组转换为PyTorch中的张量
        return idx, dist, filled_idx_n, unfilled_idx_n

    def _fill(self,holed_img, params):
        _, _ = holed_img.shape
        # print(holed_img.shape)
        # print('holed_img',holed_img)
        # holed_img.register_hook(print)
        # output=torch.zeros(size=(holed_img.shape),dtype=torch.float32,device=device,requires_grad=True)
        # output.register_hook(print)
        idx, dist, filled_idx, unfilled_idx = params
        vals = torch.zeros((1, dist.shape[0]), dtype=torch.float32, device=device)
        # print('vals.shape',vals.shape)
        # vals = torch.zeros((1, dist.shape[0]), dtype=torch.float32,device=device)
        for i in range(self.k):
            idx_select = filled_idx[idx[:, i]]  # num_coords, k
            # holed_img.register_hook(print)  # 为0
            # print( idx_select.shape)
            # print('idx_select[:, 0]',idx_select[:, 0])
            # print(idx_select[11000,0])
            # sys.exit()
            # print(holed_img[:, idx_select[:, 0], idx_select[:, 1]].shape)
            vals += holed_img[idx_select[:, 0], idx_select[:, 1]] * (1.0 / dist[:, i]) ** self.p

            # vals +=h

            # holed_img.register_hook(print) #为0
            # vals.register_hook(print) #不为0
        # holed_img.register_hook(print)
        # vals.register_hook(print) #不为0
        vals /= torch.sum((1.0 / dist) ** self.p, dim=1)
        # print('holed_img.shape',holed_img.shape)
        # vals.register_hook(print) #不为0
        # print('holed_img',holed_img) #为0
        # holed_img[:, unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        # holed_img.register_hook(print)  # 为0
        # holed_img[:,0,0]=50
        holed_img[unfilled_idx[:, 0], unfilled_idx[:, 1]] = vals
        # vals.register_hook(print) #不为0
        # holed_img.register_hook(print) #不为0
        # print('holed_img[:,unfilled_idx[:, 0], unfilled_idx[:, 1]].shape',holed_img[:,unfilled_idx[:, 0], unfilled_idx[:, 1]].shape)
        # print('holed_img1111111',holed_img)
        # return holed_img

        return holed_img
        # output[:,:]=holed_img[:,:]
        # return output




    def batch_scatter(self,input_image,device):
        # print('input_image.shape',input_image.shape)
        c,h,w=input_image.shape
        # input_image.register_hook(print)
        output=torch.zeros((c,h,w),device=device)
        for j in range(input_image.shape[0]):
            input=input_image[j,:,:]
            # input.register_hook(print) #为0
            # print('input.is_cuda',input.is_cuda) #True
            unfilled_idx = (input == 0).nonzero(as_tuple=False)
            # print('unfilled_idx',unfilled_idx) #tensor
            # unfilled_idx=unfilled_idx.detach().cpu().numpy() #.detach().cpu()
            unfilled_idx_n = unfilled_idx.cpu().numpy()
            filled_idx = (input != 0).nonzero(as_tuple=False)
            # filled_idx=filled_idx.detach().cpu().numpy() #.detach().cpu()
            filled_idx_n = filled_idx.cpu().numpy()
            # print(filled_idx)
            tree=kd.KDTree(filled_idx_n)
            dist, idx = tree.query(unfilled_idx_n, k=self.k)
            idx = idx.astype(np.int32)
            dist = torch.from_numpy(dist).to(device)  # 将numpy数组转换为PyTorch中的张量
            vals = torch.zeros((1, dist.shape[0]), dtype=torch.float32,device=device)
            for i in range(self.k):
                idx_select= filled_idx_n[idx[:, i]]

                # vals += x[idx_select[:, 0], idx_select[:, 1]]
                # w=(1.0 / dist[:, i]) ** self.p
                # print('dist[:, i].shape',dist[:, i].shape) #torch.Size([57344])
                # print('w',w)
                # print('w.shape',w.shape) #torch.Size([57344])
                # print('111',(input[idx_select[:, 0], idx_select[:, 1]]*w).shape)
                # print('idx_select[:, 0]',idx_select[:, 0])
                # print('idx_select.shape',idx_select.shape) #(57344, 2)
                # print('idx_select[:, 0].shape',idx_select[:, 0].shape) #(57344,)
                # print('idx_select[:, 1].shape',idx_select[:, 1].shape) #(57344,)
                # print('input.shape',input.shape) #torch.Size([256, 256])
                # print('222',input[idx_select[:, 0], idx_select[:, 1]])
                # print('333',input[idx_select[:, 0], idx_select[:, 1]]*3)
                # vals+=input
                # input1=input
                vals += input[idx_select[:, 0], idx_select[:, 1]]* (1.0 / dist[:, i]) ** self.p
                # vals =vals+ input[idx_select[:, 0], idx_select[:, 1]]* (1.0 / dist[:, i]) ** self.p
                # print(input1==input) #全为True
                # input.register_hook(print) #为0
                # vals.register_hook(print)#不为0
                # vals = vals+input[idx_select[:, 0], idx_select[:, 1]]* (1.0 / dist[:, i]) ** self.p
            # vals=vals/torch.sum((1.0 / dist) ** self.p, dim=1)
            vals/=torch.sum((1.0 / dist) ** self.p, dim=1)
            # vals.register_hook(print)#不为0
            # vals=vals.float()
            # print('vals.is_cuda',vals.is_cuda) #True
            # print('vals.shape',vals.shape)
            # print(type(vals[0][0]))
            # print(vals[0][0])
            # print('vals',vals)
            input[unfilled_idx_n[:, 0], unfilled_idx_n[:, 1]] = vals
            # print('unfilled_idx_n[:, 0].shape',unfilled_idx_n[:, 0].shape)
            # print('unfilled_idx_n[:, 1].shape',unfilled_idx_n[:, 1].shape)
            # print('1111111',input[unfilled_idx_n[:, 0], unfilled_idx_n[:, 1]].shape)
            # input.register_hook(print) #不为0
            # vals.register_hook(print)#不为0
            # print('input',input)
            output[j,:,:]=input
            # output[j,:,:].register_hook(print)
            # input.register_hook(print) #不为0
        # output.register_hook(print)#不为0
        # input_image.register_hook(print)
        return output
    def batch_scatter1(self,input_image,device):
        c, h, w = input_image.shape
        output = torch.zeros((c, h, w), device=device)
        '''
        
        # mask=(mask==1)
        _mask = mask[j, :, :]
        params = self._find_idx(_mask)
        filled_img = self._fill(input_image, params)
        return  filled_img
        '''
    # '''
        for j in range(input_image.shape[0]):
            input=input_image[j,:,:]
            # _mask=mask[j,:,:]
            # params=self.find_idx(_mask)
            params=self._find_idx(input)
            # input.register_hook(print) #为0
            filled_img = self._fill(input, params)
            # filled_img=self.fill(input,params)
            # filled_img.register_hook(print) #不为0
            output[j,:,:]=filled_img
        # output.register_hook(print) #不为0
        return output
    # '''

    

    def forward(self, coded): #新版本
        # print('coded.shape',coded.shape) #torch.Size([1, 4, 256, 256])
        # print('mask.shape',mask.shape) #torch.Size([4, 256, 256])
        # coded.register_hook(print)
        # print('coded',coded)
        b, c, h, w = coded.shape  # torch.Size([2, 1, 512, 512])
        # _coded=coded*1.0
        stacked = torch.zeros((b, self.num_channels, h, w), device=device)  # torch.Size([2, 4, 512, 512]))
        if self.num_channels==3 or self.num_channels==4:
            for i in range(coded.shape[0]):
                holed_img=coded[i,:,:,:]
            # '''
            # for i in range (coded.shape[1]):
            #     _mask = mask[i, :, :][None,]
            #     holed_img = coded[:, i, :, :]*_mask
            #     params = self._find_idx(_mask)
            #     filled_img = self._fill(holed_img, params)

                filled_img = self.batch_scatter1(holed_img,device)
                # params = self._find_idx(holed_img)
                # filled_img = self._fill(holed_img, params)
                # filled_img.register_hook(print) #不为0
                stacked[i, :, :, :] = filled_img
            # '''
            '''
            for i in range(coded.shape[0]):
                holed_img = coded[i, :, :, :]
                # print('holed_img',holed_img)
                # holed_img.register_hook(print) #为0
                # filled_img = self.batch_scatter(holed_img, device)
                filled_img = self.batch_scatter1(holed_img,mask, device)
                stacked[i, :, :, :] = filled_img
                # stacked.register_hook(print)
                # print('filled_img.shape',filled_img.shape) #torch.Size([4, 256, 256])
                # filled_img.register_hook(print) #不为0
            # '''
        else:
            raise NotImplementedError('this has not been implemented')
        # print('stacked.shape',stacked.shape)

        # stacked.retain_grad()
        # stacked.register_hook(print) #不为0
        # print(stacked.register_hook(print)==0)
        # print('coded',coded)
        # print('stacked',stacked)
        # print(coded==stacked)
        return stacked

# class Batch_scatter(nn.Module):
#     def __init__(self,):
#
#     def forward(self,):
