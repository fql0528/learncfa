import torch
import torch.nn as nn
import os
import shutters.shutter_utils as shutils
import torch.nn.functional as F
import numpy as np
import  math
from Proxy.Proxy_function import Proxy_function
import sys
# -*- coding: gbk -*-
from Soft_max.My_Gumbel_Softmax import my_gumbel_softmax
from torchvision.utils import save_image
import torch.nn.functional  as F

if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(precision=8)

def add_noise(img, exp_time=1, test=False):
    sig_shot_min = 0.0001
    sig_shot_max = 0.01
    sig_read_min = 0.001
    sig_read_max = 0.03
    if test: # keep noise levels fixed when testing
        sig_shot = (sig_shot_max - sig_shot_min) / 2
        sig_read = (sig_read_max - sig_read_min) / 2
    else:
        sig_shot = (sig_shot_min - sig_shot_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_shot_max
        sig_read = (sig_read_min - sig_read_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_read_max

    ratio = exp_time / 8

    # Scale image corresponding to exposure time
    img = img * ratio

    # Add shot noise, must detach or it'll mess with the computational graph
    shot = (img.detach() ** (1/2)) * sig_shot * torch.randn_like(img)

    # Add read noise. Short and long exposures should have the same read noise.
    read = sig_read * torch.randn_like(img)
    return img + shot + read

class Shutter:
    def __new__(cls, shutter_type,block_size,w_zhi,alpha,cfa_size,test=False, resume=False,init='even'):
        cls_out = {
            'bayerrgb':BAYERRGB,
            'gindelergbw':GINDELERGBW,
            'luorgbw':LUORGBW,
            'wangrgbw':WANGRGBW,
            'cfzrgbw':CFZRGBW,
            'nipsrgbw':NIPSRGBW,
            'binningrgbw': BINNINGRGBW,
            'sonyrgbw': SONYRGBW,
            'yamagamirgbw':YAMAGAMIRGBW,
            'kaizurgbw':KAIZURGBW,
            'hamiltonrgbw': HAMILTONRGBW,
            'hondargbw': HONDARGBW,
            'randomrgbw': RANDOMRGBW,
            'ourrgbw':OURRGBW,
            'rgbw':RGBW,
            'lrgbw':LRGBW,
        }[shutter_type]

        return cls_out(block_size,cfa_size,w_zhi,alpha,test, resume, init)

class ShutterBase(nn.Module):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False, init='even'):
        super().__init__()
        self.block_size = block_size
        self.test = test
        self.resume = resume
        self.cfa_size=cfa_size
        # self.model_dir = os.path.dirname(model_dir) # '21-11-14/21-11-14-net/short'

    def getLength(self):
        raise NotImplementedError('Must implement in derived class')

    def getMeasurementMatrix(self):
        raise NotImplementedError('Must implement in derived class')

    def forward(self, video_block):
        raise NotImplementedError('Must implement in derived class')

    def post_process(self, measurement, exp_time, test):  #exp_time：表示曝光时间
        # measurement.shape torch.Size([2, 1, 512, 512])
        #  exp_time.shape  torch.Size([1,512,512])
        measurement = torch.div(measurement, exp_time)       # 1 1 H W
        # print('measurement.shape',measurement.shape) #torch.Size([1, 4, 186, 317])

        measurement = add_noise(measurement, exp_time=exp_time, test=test) #原版

        measurement = torch.clamp(measurement, 0, 1)  # 把measurement归一化到0--1
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        return measurement

    def Proxy_function(self,cfa_image, cfa, init, image, w, mask):
        # cfa_image=torch.zeros(size=(image.shape),device=device)
        # cfa=torch.zeros(size=(image.shape[1:]),device=device)
        if init == 'softmax_tau':
            # 带温度系数的softmax函数
            # print('代理函数：带温度系数的softmax函数')
            x = shutils.produce_rgbw_cfa(w)
            # x.register_hook(print)
            # print('x.shape',x.shape)
            # print('x',x)
            for i in range(4):
                cfa[i] = torch.sum(x[:, i] * mask[:, i], dim=0, keepdim=True)
                cfa_image[:, i] = cfa[i] * image[:, i]
                # cfa_image.register_hook(print)
            return cfa_image, cfa
        elif init == 'gumbel_softmax_tau_1':
            # Gumbel_Softmax
            print('代理函数：Gumbel_Softmax函数,tau=1')
            tau = 1.0
            # x = F.gumbel_softmax(logits=w, hard=True, tau=tau, dim=1)
            x=my_gumbel_softmax(logits=w, hard=True, tau=tau, dim=1) #未加gumbel分布
            # print('x',x)
            # exit()
            for i in range(4):
                cfa[i] = torch.sum(x[:, i] * mask[:, i], dim=0, keepdim=True)
                cfa_image[:, i] = cfa[i] * image[:, i]
            # print(cfa)
            # exit()
            return cfa_image, cfa
        elif init == 'gumbel_softmax_tau':
            # Gumbel_SoftMax
            print("代理函数：Gumbel_Softmax函数,tau可变")
            t0 = 10  # 初始温度
            # T=1 #迭代次数
            global T  # 迭代次数
            tau = t0 / (1 + math.log(T))
            x = F.gumbel_softmax(logits=w, hard=True, tau=tau, dim=1)
            for i in range(4):
                cfa[i] = torch.sum(x[:, i] * mask[:, i], dim=0, keepdim=True)
                cfa_image[:, i] = cfa[i] * image[:, i]
            T += 1
            # print('T',T)
            return cfa_image, cfa
        else:
            print("选择了未知代理函数,程序error")
            sys.exit(-1)

    def count_instances(self, lengths, counts):
        # print('lengths',lengths)
        # print('lengths.shape',lengths.shape) #torch.Size([186, 317])
        # print('counts',counts) #counts {1: 0, 2: 0, 3: 0, 4: 0}
        flattened_lengths = lengths.reshape(-1, ).type(torch.int8)  #展平的函数
        # print('flattened_lengths',flattened_lengths) #tensor([2, 4, 1,  ..., 4, 1, 4], dtype=torch.int8)
        # print('flattened_lengths.shape',flattened_lengths.shape) #torch.Size([58962])
        total_counts = torch.bincount(flattened_lengths).cpu()
        # print('total_counts',total_counts)#tensor([    0,  7426, 14741,  7314, 29481])
        for k in range(1, len(total_counts)):
            counts[k] = total_counts[k]
        # print('counts',counts) #counts {1: tensor(7426), 2: tensor(14741), 3: tensor(7314), 4: tensor(29481)}
        return counts

class RGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
         # 柯达 4x4CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # red channels
        self.cfa_256x256[0,::4,2::4]=1
        self.cfa_256x256[0,1::4,3::4]=1
        #green channels
        self.cfa_256x256[1,::4,::4]=1
        self.cfa_256x256[1,1::4,1::4]=1
        self.cfa_256x256[1,2::4,2::4]=1
        self.cfa_256x256[1,3::4,3::4]=1
        #blue channels
        self.cfa_256x256[2,2::4,::4]=1
        self.cfa_256x256[2,3::4,1::4]=1
        #W channel
        self.cfa_256x256[3,::2,1::2]=1
        self.cfa_256x256[3,1::2,::2]=1
        # """

        """
        # 2016 NIPS论文8x8CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # 第一行
        self.cfa_256x256[3,0::8,0::8]=1
        self.cfa_256x256[1,0::8,1::8]=1
        self.cfa_256x256[3,0::8,2::8]=1
        self.cfa_256x256[3,0::8,3::8]=1
        self.cfa_256x256[0,0::8,4::8]=1
        self.cfa_256x256[3,0::8,5::8]=1
        self.cfa_256x256[3,0::8,6::8]=1
        self.cfa_256x256[3,0::8,7::8]=1
        # 第二行
        self.cfa_256x256[0,1::8,0::8]=1
        self.cfa_256x256[3,1::8,1::8]=1
        self.cfa_256x256[3,1::8,2::8]=1
        self.cfa_256x256[3,1::8,3::8]=1
        self.cfa_256x256[3,1::8,4::8]=1
        self.cfa_256x256[3,1::8,5::8]=1
        self.cfa_256x256[0,1::8,6::8]=1
        self.cfa_256x256[3,1::8,7::8]=1
        #第三行
        self.cfa_256x256[1,2::8,0::8]=1
        self.cfa_256x256[3,2::8,1::8]=1
        self.cfa_256x256[2,2::8,2::8]=1
        self.cfa_256x256[2,2::8,3::8]=1
        self.cfa_256x256[3,2::8,4::8]=1
        self.cfa_256x256[2,2::8,5::8]=1
        self.cfa_256x256[3,2::8,6::8]=1
        self.cfa_256x256[3,2::8,7::8]=1
        # 第四行
        self.cfa_256x256[0,3::8,0::8]=1
        self.cfa_256x256[2,3::8,1::8]=1
        self.cfa_256x256[3,3::8,2::8]=1
        self.cfa_256x256[3,3::8,3::8]=1
        self.cfa_256x256[1,3::8,4::8]=1
        self.cfa_256x256[3,3::8,5::8]=1
        self.cfa_256x256[1,3::8,6::8]=1
        self.cfa_256x256[2,3::8,7::8]=1
        # 第五行
        self.cfa_256x256[3,4::8,0::8]=1
        self.cfa_256x256[3,4::8,1::8]=1
        self.cfa_256x256[3,4::8,2::8]=1
        self.cfa_256x256[3,4::8,3::8]=1
        self.cfa_256x256[3,4::8,4::8]=1
        self.cfa_256x256[1,4::8,5::8]=1
        self.cfa_256x256[0,4::8,6::8]=1
        self.cfa_256x256[3,4::8,7::8]=1
        # 第六行
        self.cfa_256x256[3,5::8,0::8]=1
        self.cfa_256x256[0,5::8,1::8]=1
        self.cfa_256x256[3,5::8,2::8]=1
        self.cfa_256x256[1,5::8,3::8]=1
        self.cfa_256x256[3,5::8,4::8]=1
        self.cfa_256x256[3,5::8,5::8]=1
        self.cfa_256x256[1,5::8,6::8]=1
        self.cfa_256x256[0,5::8,7::8]=1
        # 第七行
        self.cfa_256x256[0,6::8,0::8]=1
        self.cfa_256x256[3,6::8,1::8]=1
        self.cfa_256x256[2,6::8,2::8]=1
        self.cfa_256x256[3,6::8,3::8]=1
        self.cfa_256x256[1,6::8,4::8]=1
        self.cfa_256x256[3,6::8,5::8]=1
        self.cfa_256x256[3,6::8,6::8]=1
        self.cfa_256x256[3,6::8,7::8]=1
        # 第八行
        self.cfa_256x256[3,7::8,0::8]=1
        self.cfa_256x256[1,7::8,1::8]=1
        self.cfa_256x256[3,7::8,2::8]=1
        self.cfa_256x256[3,7::8,3::8]=1
        self.cfa_256x256[3,7::8,4::8]=1
        self.cfa_256x256[3,7::8,5::8]=1
        self.cfa_256x256[2,7::8,6::8]=1
        self.cfa_256x256[2,7::8,7::8]=1
        """

    def forward(self,img_rgbw,train=True):
        
        if train==True:
            mask=self.cfa_256x256
           
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class BAYERRGB(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        self.bayercfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        #Red channels
        self.bayercfa_256x256[0,::2,1::2]=1
        #Green channels
        self.bayercfa_256x256[1,::2,::2]=1
        self.bayercfa_256x256[1,1::2,1::2]=1
         #Blue channels
        self.bayercfa_256x256[2,1::2,::2]=1
        
    def forward(self,img_rgb,train=True):
        if train==True:
            mask=self.bayercfa_256x256
            # print('mask',mask[:,:5,:5])
            # save_image(mask[:3,:9,:9], 'bayercfa.png')
            # exit()
            # bayer_rgbw=mask*img_rgbw
            # exit()
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.bayercfa_256x256,(1,np.where(img_rgb.shape[2]%self.bayercfa_256x256.shape[1]==0,img_rgb.shape[2]//self.bayercfa_256x256.shape[1],img_rgb.shape[2]//self.bayercfa_256x256.shape[1]+1),np.where(img_rgb.shape[3]%self.bayercfa_256x256.shape[2]==0,img_rgb.shape[3]//self.bayercfa_256x256.shape[2],img_rgb.shape[3]//self.bayercfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgb.shape[2],:img_rgb.shape[3]]
        bayer_rgb=mask*img_rgb
        return bayer_rgb,mask
    
class NIPSRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        # 2016 NIPS论文8x8CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # 第一行
        self.cfa_256x256[3,0::8,0::8]=1
        self.cfa_256x256[1,0::8,1::8]=1
        self.cfa_256x256[3,0::8,2::8]=1
        self.cfa_256x256[3,0::8,3::8]=1
        self.cfa_256x256[0,0::8,4::8]=1
        self.cfa_256x256[3,0::8,5::8]=1
        self.cfa_256x256[3,0::8,6::8]=1
        self.cfa_256x256[3,0::8,7::8]=1
        # 第二行
        self.cfa_256x256[0,1::8,0::8]=1
        self.cfa_256x256[3,1::8,1::8]=1
        self.cfa_256x256[3,1::8,2::8]=1
        self.cfa_256x256[3,1::8,3::8]=1
        self.cfa_256x256[3,1::8,4::8]=1
        self.cfa_256x256[3,1::8,5::8]=1
        self.cfa_256x256[0,1::8,6::8]=1
        self.cfa_256x256[3,1::8,7::8]=1
        #第三行
        self.cfa_256x256[1,2::8,0::8]=1
        self.cfa_256x256[3,2::8,1::8]=1
        self.cfa_256x256[2,2::8,2::8]=1
        self.cfa_256x256[2,2::8,3::8]=1
        self.cfa_256x256[3,2::8,4::8]=1
        self.cfa_256x256[2,2::8,5::8]=1
        self.cfa_256x256[3,2::8,6::8]=1
        self.cfa_256x256[3,2::8,7::8]=1
        # 第四行
        self.cfa_256x256[0,3::8,0::8]=1
        self.cfa_256x256[2,3::8,1::8]=1
        self.cfa_256x256[3,3::8,2::8]=1
        self.cfa_256x256[3,3::8,3::8]=1
        self.cfa_256x256[1,3::8,4::8]=1
        self.cfa_256x256[3,3::8,5::8]=1
        self.cfa_256x256[1,3::8,6::8]=1
        self.cfa_256x256[2,3::8,7::8]=1
        # 第五行
        self.cfa_256x256[3,4::8,0::8]=1
        self.cfa_256x256[3,4::8,1::8]=1
        self.cfa_256x256[3,4::8,2::8]=1
        self.cfa_256x256[3,4::8,3::8]=1
        self.cfa_256x256[3,4::8,4::8]=1
        self.cfa_256x256[1,4::8,5::8]=1
        self.cfa_256x256[0,4::8,6::8]=1
        self.cfa_256x256[3,4::8,7::8]=1
        # 第六行
        self.cfa_256x256[3,5::8,0::8]=1
        self.cfa_256x256[0,5::8,1::8]=1
        self.cfa_256x256[3,5::8,2::8]=1
        self.cfa_256x256[1,5::8,3::8]=1
        self.cfa_256x256[3,5::8,4::8]=1
        self.cfa_256x256[3,5::8,5::8]=1
        self.cfa_256x256[1,5::8,6::8]=1
        self.cfa_256x256[0,5::8,7::8]=1
        # 第七行
        self.cfa_256x256[0,6::8,0::8]=1
        self.cfa_256x256[3,6::8,1::8]=1
        self.cfa_256x256[2,6::8,2::8]=1
        self.cfa_256x256[3,6::8,3::8]=1
        self.cfa_256x256[1,6::8,4::8]=1
        self.cfa_256x256[3,6::8,5::8]=1
        self.cfa_256x256[3,6::8,6::8]=1
        self.cfa_256x256[3,6::8,7::8]=1
        # 第八行
        self.cfa_256x256[3,7::8,0::8]=1
        self.cfa_256x256[1,7::8,1::8]=1
        self.cfa_256x256[3,7::8,2::8]=1
        self.cfa_256x256[3,7::8,3::8]=1
        self.cfa_256x256[3,7::8,4::8]=1
        self.cfa_256x256[3,7::8,5::8]=1
        self.cfa_256x256[2,7::8,6::8]=1
        self.cfa_256x256[2,7::8,7::8]=1
        # """

    def forward(self,img_rgbw,train=True):
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class HAMILTONRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        # 2016 NIPS论文8x8CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # 第一行
        self.cfa_256x256[0,0::8,0::8]=1
        self.cfa_256x256[3,0::8,1::8]=1
        self.cfa_256x256[0,0::8,2::8]=1
        self.cfa_256x256[3,0::8,3::8]=1
        self.cfa_256x256[1,0::8,4::8]=1
        self.cfa_256x256[3,0::8,5::8]=1
        self.cfa_256x256[1,0::8,6::8]=1
        self.cfa_256x256[3,0::8,7::8]=1
        # 第二行
        self.cfa_256x256[3,1::8,0::8]=1
        self.cfa_256x256[0,1::8,1::8]=1
        self.cfa_256x256[3,1::8,2::8]=1
        self.cfa_256x256[0,1::8,3::8]=1
        self.cfa_256x256[3,1::8,4::8]=1
        self.cfa_256x256[1,1::8,5::8]=1
        self.cfa_256x256[3,1::8,6::8]=1
        self.cfa_256x256[1,1::8,7::8]=1
        #第三行
        self.cfa_256x256[0,2::8,0::8]=1
        self.cfa_256x256[3,2::8,1::8]=1
        self.cfa_256x256[0,2::8,2::8]=1
        self.cfa_256x256[3,2::8,3::8]=1
        self.cfa_256x256[1,2::8,4::8]=1
        self.cfa_256x256[3,2::8,5::8]=1
        self.cfa_256x256[1,2::8,6::8]=1
        self.cfa_256x256[3,2::8,7::8]=1
        # 第四行
        self.cfa_256x256[3,3::8,0::8]=1
        self.cfa_256x256[0,3::8,1::8]=1
        self.cfa_256x256[3,3::8,2::8]=1
        self.cfa_256x256[0,3::8,3::8]=1
        self.cfa_256x256[3,3::8,4::8]=1
        self.cfa_256x256[1,3::8,5::8]=1
        self.cfa_256x256[3,3::8,6::8]=1
        self.cfa_256x256[1,3::8,7::8]=1
        # 第五行
        self.cfa_256x256[1,4::8,0::8]=1
        self.cfa_256x256[3,4::8,1::8]=1
        self.cfa_256x256[1,4::8,2::8]=1
        self.cfa_256x256[3,4::8,3::8]=1
        self.cfa_256x256[2,4::8,4::8]=1
        self.cfa_256x256[3,4::8,5::8]=1
        self.cfa_256x256[2,4::8,6::8]=1
        self.cfa_256x256[3,4::8,7::8]=1
        # 第六行
        self.cfa_256x256[3,5::8,0::8]=1
        self.cfa_256x256[1,5::8,1::8]=1
        self.cfa_256x256[3,5::8,2::8]=1
        self.cfa_256x256[1,5::8,3::8]=1
        self.cfa_256x256[3,5::8,4::8]=1
        self.cfa_256x256[2,5::8,5::8]=1
        self.cfa_256x256[3,5::8,6::8]=1
        self.cfa_256x256[2,5::8,7::8]=1
        # 第七行
        self.cfa_256x256[1,6::8,0::8]=1
        self.cfa_256x256[3,6::8,1::8]=1
        self.cfa_256x256[1,6::8,2::8]=1
        self.cfa_256x256[3,6::8,3::8]=1
        self.cfa_256x256[2,6::8,4::8]=1
        self.cfa_256x256[3,6::8,5::8]=1
        self.cfa_256x256[2,6::8,6::8]=1
        self.cfa_256x256[3,6::8,7::8]=1
        # 第八行
        self.cfa_256x256[3,7::8,0::8]=1
        self.cfa_256x256[1,7::8,1::8]=1
        self.cfa_256x256[3,7::8,2::8]=1
        self.cfa_256x256[1,7::8,3::8]=1
        self.cfa_256x256[3,7::8,4::8]=1
        self.cfa_256x256[2,7::8,5::8]=1
        self.cfa_256x256[3,7::8,6::8]=1
        self.cfa_256x256[2,7::8,7::8]=1
        # """

    def forward(self,img_rgbw,train=True):
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask


class RANDOMRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        # 2016 NIPS论文8x8CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # 第一行
        self.cfa_256x256[3,0::8,0::8]=1
        self.cfa_256x256[3,0::8,1::8]=1
        self.cfa_256x256[3,0::8,2::8]=1
        self.cfa_256x256[3,0::8,3::8]=1
        self.cfa_256x256[3,0::8,4::8]=1
        self.cfa_256x256[3,0::8,5::8]=1
        self.cfa_256x256[3,0::8,6::8]=1
        self.cfa_256x256[3,0::8,7::8]=1
        # 第二行
        self.cfa_256x256[3,1::8,0::8]=1
        self.cfa_256x256[2,1::8,1::8]=1
        self.cfa_256x256[3,1::8,2::8]=1
        self.cfa_256x256[1,1::8,3::8]=1
        self.cfa_256x256[3,1::8,4::8]=1
        self.cfa_256x256[2,1::8,5::8]=1
        self.cfa_256x256[3,1::8,6::8]=1
        self.cfa_256x256[0,1::8,7::8]=1
        #第三行
        self.cfa_256x256[3,2::8,0::8]=1
        self.cfa_256x256[3,2::8,1::8]=1
        self.cfa_256x256[3,2::8,2::8]=1
        self.cfa_256x256[3,2::8,3::8]=1
        self.cfa_256x256[3,2::8,4::8]=1
        self.cfa_256x256[3,2::8,5::8]=1
        self.cfa_256x256[3,2::8,6::8]=1
        self.cfa_256x256[3,2::8,7::8]=1
        # 第四行
        self.cfa_256x256[3,3::8,0::8]=1
        self.cfa_256x256[0,3::8,1::8]=1
        self.cfa_256x256[3,3::8,2::8]=1
        self.cfa_256x256[0,3::8,3::8]=1
        self.cfa_256x256[3,3::8,4::8]=1
        self.cfa_256x256[2,3::8,5::8]=1
        self.cfa_256x256[3,3::8,6::8]=1
        self.cfa_256x256[1,3::8,7::8]=1
        # 第五行
        self.cfa_256x256[3,4::8,0::8]=1
        self.cfa_256x256[3,4::8,1::8]=1
        self.cfa_256x256[3,4::8,2::8]=1
        self.cfa_256x256[3,4::8,3::8]=1
        self.cfa_256x256[3,4::8,4::8]=1
        self.cfa_256x256[3,4::8,5::8]=1
        self.cfa_256x256[3,4::8,6::8]=1
        self.cfa_256x256[3,4::8,7::8]=1
        # 第六行
        self.cfa_256x256[3,5::8,0::8]=1
        self.cfa_256x256[2,5::8,1::8]=1
        self.cfa_256x256[3,5::8,2::8]=1
        self.cfa_256x256[1,5::8,3::8]=1
        self.cfa_256x256[3,5::8,4::8]=1
        self.cfa_256x256[1,5::8,5::8]=1
        self.cfa_256x256[3,5::8,6::8]=1
        self.cfa_256x256[0,5::8,7::8]=1
        # 第七行
        self.cfa_256x256[3,6::8,0::8]=1
        self.cfa_256x256[3,6::8,1::8]=1
        self.cfa_256x256[3,6::8,2::8]=1
        self.cfa_256x256[3,6::8,3::8]=1
        self.cfa_256x256[3,6::8,4::8]=1
        self.cfa_256x256[3,6::8,5::8]=1
        self.cfa_256x256[3,6::8,6::8]=1
        self.cfa_256x256[3,6::8,7::8]=1
        # 第八行
        self.cfa_256x256[3,7::8,0::8]=1
        self.cfa_256x256[0,7::8,1::8]=1
        self.cfa_256x256[3,7::8,2::8]=1
        self.cfa_256x256[2,7::8,3::8]=1
        self.cfa_256x256[3,7::8,4::8]=1
        self.cfa_256x256[2,7::8,5::8]=1
        self.cfa_256x256[3,7::8,6::8]=1
        self.cfa_256x256[1,7::8,7::8]=1
        # """

    def forward(self,img_rgbw,train=True):
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class KAIZURGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        # Kai zu CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # 第一行
        self.cfa_256x256[0,0::6,0::6]=1
        self.cfa_256x256[3,0::6,1::6]=1
        self.cfa_256x256[1,0::6,2::6]=1
        self.cfa_256x256[3,0::6,3::6]=1
        self.cfa_256x256[2,0::6,4::6]=1
        self.cfa_256x256[3,0::6,5::6]=1
        # 第二行
        self.cfa_256x256[3,1::6,0::6]=1
        self.cfa_256x256[0,1::6,1::6]=1
        self.cfa_256x256[3,1::6,2::6]=1
        self.cfa_256x256[1,1::6,3::6]=1
        self.cfa_256x256[3,1::6,4::6]=1
        self.cfa_256x256[2,1::6,5::6]=1
        #第三行
        self.cfa_256x256[1,2::6,0::6]=1
        self.cfa_256x256[3,2::6,1::6]=1
        self.cfa_256x256[2,2::6,2::6]=1
        self.cfa_256x256[3,2::6,3::6]=1
        self.cfa_256x256[0,2::6,4::6]=1
        self.cfa_256x256[3,2::6,5::6]=1
        # 第四行
        self.cfa_256x256[3,3::6,0::6]=1
        self.cfa_256x256[1,3::6,1::6]=1
        self.cfa_256x256[3,3::6,2::6]=1
        self.cfa_256x256[2,3::6,3::6]=1
        self.cfa_256x256[3,3::6,4::6]=1
        self.cfa_256x256[0,3::6,5::6]=1
        # 第五行
        self.cfa_256x256[2,4::6,0::6]=1
        self.cfa_256x256[3,4::6,1::6]=1
        self.cfa_256x256[0,4::6,2::6]=1
        self.cfa_256x256[3,4::6,3::6]=1
        self.cfa_256x256[1,4::6,4::6]=1
        self.cfa_256x256[3,4::6,5::6]=1
        # 第六行
        self.cfa_256x256[3,5::6,0::6]=1
        self.cfa_256x256[2,5::6,1::6]=1
        self.cfa_256x256[3,5::6,2::6]=1
        self.cfa_256x256[0,5::6,3::6]=1
        self.cfa_256x256[3,5::6,4::6]=1
        self.cfa_256x256[1,5::6,5::6]=1

    def forward(self,img_rgbw,train=True):
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class WANGRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        #  CFA
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
        # 第一行
        self.cfa_256x256[3,0::5,0::5]=1
        self.cfa_256x256[0,0::5,1::5]=1
        self.cfa_256x256[2,0::5,2::5]=1
        self.cfa_256x256[3,0::5,3::5]=1
        self.cfa_256x256[1,0::5,4::5]=1
        # 第二行
        self.cfa_256x256[3,1::5,0::5]=1
        self.cfa_256x256[1,1::5,1::5]=1
        self.cfa_256x256[3,1::5,2::5]=1
        self.cfa_256x256[0,1::5,3::5]=1
        self.cfa_256x256[2,1::5,4::5]=1
        #第三行
        self.cfa_256x256[0,2::5,0::5]=1
        self.cfa_256x256[2,2::5,1::5]=1
        self.cfa_256x256[3,2::5,2::5]=1
        self.cfa_256x256[1,2::5,3::5]=1
        self.cfa_256x256[3,2::5,4::5]=1
        # 第四行
        self.cfa_256x256[1,3::5,0::5]=1
        self.cfa_256x256[3,3::5,1::5]=1
        self.cfa_256x256[0,3::5,2::5]=1
        self.cfa_256x256[2,3::5,3::5]=1
        self.cfa_256x256[3,3::5,4::5]=1
        # 第五行
        self.cfa_256x256[2,4::5,0::5]=1
        self.cfa_256x256[3,4::5,1::5]=1
        self.cfa_256x256[1,4::5,2::5]=1
        self.cfa_256x256[3,4::5,3::5]=1
        self.cfa_256x256[0,4::5,4::5]=1

    def forward(self,img_rgbw,train=True):
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class OURRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # Ours2 4x4CFA
        # 第一行
        self.cfa_256x256[2,0::4,0::4]=1
        self.cfa_256x256[0,0::4,1::4]=1
        self.cfa_256x256[3,0::4,2::4]=1
        self.cfa_256x256[1,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[1,1::4,0::4]=1
        self.cfa_256x256[3,1::4,1::4]=1
        self.cfa_256x256[1,1::4,2::4]=1
        self.cfa_256x256[0,1::4,3::4]=1
        #第三行
        self.cfa_256x256[2,2::4,0::4]=1
        self.cfa_256x256[3,2::4,1::4]=1
        self.cfa_256x256[1,2::4,2::4]=1
        self.cfa_256x256[3,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[1,3::4,0::4]=1
        self.cfa_256x256[3,3::4,1::4]=1
        self.cfa_256x256[1,3::4,2::4]=1
        self.cfa_256x256[1,3::4,3::4]=1
        # """
        
        """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # Ours3 4x4CFA
        # 第一行
        self.cfa_256x256[0,0::4,0::4]=1
        self.cfa_256x256[3,0::4,1::4]=1
        self.cfa_256x256[2,0::4,2::4]=1
        self.cfa_256x256[1,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[3,1::4,0::4]=1
        self.cfa_256x256[2,1::4,1::4]=1
        self.cfa_256x256[3,1::4,2::4]=1
        self.cfa_256x256[3,1::4,3::4]=1
        #第三行
        self.cfa_256x256[2,2::4,0::4]=1
        self.cfa_256x256[1,2::4,1::4]=1
        self.cfa_256x256[2,2::4,2::4]=1
        self.cfa_256x256[1,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[0,3::4,0::4]=1
        self.cfa_256x256[1,3::4,1::4]=1
        self.cfa_256x256[0,3::4,2::4]=1
        self.cfa_256x256[3,3::4,3::4]=1
        """

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class HONDARGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # Ours 4x4CFA
        # 第一行
        self.cfa_256x256[3,0::4,0::4]=1
        self.cfa_256x256[3,0::4,1::4]=1
        self.cfa_256x256[3,0::4,2::4]=1
        self.cfa_256x256[3,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[3,1::4,0::4]=1
        self.cfa_256x256[1,1::4,1::4]=1
        self.cfa_256x256[3,1::4,2::4]=1
        self.cfa_256x256[0,1::4,3::4]=1
        #第三行
        self.cfa_256x256[3,2::4,0::4]=1
        self.cfa_256x256[3,2::4,1::4]=1
        self.cfa_256x256[3,2::4,2::4]=1
        self.cfa_256x256[3,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[3,3::4,0::4]=1
        self.cfa_256x256[2,3::4,1::4]=1
        self.cfa_256x256[3,3::4,2::4]=1
        self.cfa_256x256[1,3::4,3::4]=1

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask


class GINDELERGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # 4x4CFA
        # 第一行
        self.cfa_256x256[1,0::4,0::4]=1
        self.cfa_256x256[0,0::4,1::4]=1
        self.cfa_256x256[1,0::4,2::4]=1
        self.cfa_256x256[0,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[2,1::4,0::4]=1
        self.cfa_256x256[3,1::4,1::4]=1
        self.cfa_256x256[2,1::4,2::4]=1
        self.cfa_256x256[3,1::4,3::4]=1
        #第三行
        self.cfa_256x256[1,2::4,0::4]=1
        self.cfa_256x256[0,2::4,1::4]=1
        self.cfa_256x256[1,2::4,2::4]=1
        self.cfa_256x256[0,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[2,3::4,0::4]=1
        self.cfa_256x256[3,3::4,1::4]=1
        self.cfa_256x256[2,3::4,2::4]=1
        self.cfa_256x256[3,3::4,3::4]=1

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask
    

class LUORGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # 4x4CFA
        # 第一行
        self.cfa_256x256[3,0::4,0::4]=1
        self.cfa_256x256[1,0::4,1::4]=1
        self.cfa_256x256[3,0::4,2::4]=1
        self.cfa_256x256[3,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[0,1::4,0::4]=1
        self.cfa_256x256[2,1::4,1::4]=1
        self.cfa_256x256[0,1::4,2::4]=1
        self.cfa_256x256[3,1::4,3::4]=1
        #第三行
        self.cfa_256x256[3,2::4,0::4]=1
        self.cfa_256x256[1,2::4,1::4]=1
        self.cfa_256x256[3,2::4,2::4]=1
        self.cfa_256x256[3,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[3,3::4,0::4]=1
        self.cfa_256x256[3,3::4,1::4]=1
        self.cfa_256x256[3,3::4,2::4]=1
        self.cfa_256x256[3,3::4,3::4]=1

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class BINNINGRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # 4x4CFA
        # 第一行
        self.cfa_256x256[1,0::4,0::4]=1
        self.cfa_256x256[1,0::4,1::4]=1
        self.cfa_256x256[0,0::4,2::4]=1
        self.cfa_256x256[0,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[1,1::4,0::4]=1
        self.cfa_256x256[1,1::4,1::4]=1
        self.cfa_256x256[0,1::4,2::4]=1
        self.cfa_256x256[0,1::4,3::4]=1
        #第三行
        self.cfa_256x256[2,2::4,0::4]=1
        self.cfa_256x256[2,2::4,1::4]=1
        self.cfa_256x256[3,2::4,2::4]=1
        self.cfa_256x256[3,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[2,3::4,0::4]=1
        self.cfa_256x256[2,3::4,1::4]=1
        self.cfa_256x256[3,3::4,2::4]=1
        self.cfa_256x256[3,3::4,3::4]=1
        # save_image(self.cfa_256x256[:3,:6,:6],'xx.png')

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask


class SONYRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # 4x4CFA
        # 第一行
        self.cfa_256x256[3,0::4,0::4]=1
        self.cfa_256x256[0,0::4,1::4]=1
        self.cfa_256x256[3,0::4,2::4]=1
        self.cfa_256x256[1,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[2,1::4,0::4]=1
        self.cfa_256x256[3,1::4,1::4]=1
        self.cfa_256x256[1,1::4,2::4]=1
        self.cfa_256x256[3,1::4,3::4]=1
        #第三行
        self.cfa_256x256[3,2::4,0::4]=1
        self.cfa_256x256[1,2::4,1::4]=1
        self.cfa_256x256[3,2::4,2::4]=1
        self.cfa_256x256[0,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[1,3::4,0::4]=1
        self.cfa_256x256[3,3::4,1::4]=1
        self.cfa_256x256[2,3::4,2::4]=1
        self.cfa_256x256[3,3::4,3::4]=1

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class YAMAGAMIRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # 4x4CFA
        # 第一行
        self.cfa_256x256[2,0::4,0::4]=1
        self.cfa_256x256[3,0::4,1::4]=1
        self.cfa_256x256[0,0::4,2::4]=1
        self.cfa_256x256[3,0::4,3::4]=1
        # 第二行
        self.cfa_256x256[3,1::4,0::4]=1
        self.cfa_256x256[1,1::4,1::4]=1
        self.cfa_256x256[3,1::4,2::4]=1
        self.cfa_256x256[1,1::4,3::4]=1
        #第三行
        self.cfa_256x256[0,2::4,0::4]=1
        self.cfa_256x256[3,2::4,1::4]=1
        self.cfa_256x256[2,2::4,2::4]=1
        self.cfa_256x256[3,2::4,3::4]=1
        # 第四行
        self.cfa_256x256[3,3::4,0::4]=1
        self.cfa_256x256[1,3::4,1::4]=1
        self.cfa_256x256[3,3::4,2::4]=1
        self.cfa_256x256[1,3::4,3::4]=1

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask


class CFZRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size,w_zhi,alpha,test=False, resume=False,init='even'):
        super().__init__( block_size,cfa_size,w_zhi,alpha,test, resume,init)
        # """
        self.cfa_256x256=torch.zeros(size=(block_size[0],block_size[1],block_size[2]),dtype=torch.float32,device=device)
         # CFZ RGBW CFA
        # 第一行
        self.cfa_256x256[3,0::6,0::6]=1
        self.cfa_256x256[3,0::6,1::6]=1
        self.cfa_256x256[3,0::6,2::6]=1
        self.cfa_256x256[3,0::6,3::6]=1
        self.cfa_256x256[3,0::6,4::6]=1
        self.cfa_256x256[3,0::6,5::6]=1
        # 第二行
        self.cfa_256x256[3,1::6,0::6]=1
        self.cfa_256x256[3,1::6,1::6]=1
        self.cfa_256x256[3,1::6,2::6]=1
        self.cfa_256x256[3,1::6,3::6]=1
        self.cfa_256x256[3,1::6,4::6]=1
        self.cfa_256x256[3,1::6,5::6]=1
        #第三行
        self.cfa_256x256[3,2::6,0::6]=1
        self.cfa_256x256[3,2::6,1::6]=1
        self.cfa_256x256[1,2::6,2::6]=1
        self.cfa_256x256[0,2::6,3::6]=1
        self.cfa_256x256[3,2::6,4::6]=1
        self.cfa_256x256[3,2::6,5::6]=1
        # 第四行
        self.cfa_256x256[3,3::6,0::6]=1
        self.cfa_256x256[3,3::6,1::6]=1
        self.cfa_256x256[2,3::6,2::6]=1
        self.cfa_256x256[1,3::6,3::6]=1
        self.cfa_256x256[3,3::6,4::6]=1
        self.cfa_256x256[3,3::6,5::6]=1
        # 第五行
        self.cfa_256x256[3,4::6,0::6]=1
        self.cfa_256x256[3,4::6,1::6]=1
        self.cfa_256x256[3,4::6,2::6]=1
        self.cfa_256x256[3,4::6,3::6]=1
        self.cfa_256x256[3,4::6,4::6]=1
        self.cfa_256x256[3,4::6,5::6]=1
        # 第六行
        self.cfa_256x256[3,5::6,0::6]=1
        self.cfa_256x256[3,5::6,1::6]=1
        self.cfa_256x256[3,5::6,2::6]=1
        self.cfa_256x256[3,5::6,3::6]=1
        self.cfa_256x256[3,5::6,4::6]=1
        self.cfa_256x256[3,5::6,5::6]=1

    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        if train==True:
            mask=self.cfa_256x256
        elif train==False:
            # mask=np.tile(self.cfa_256x256.numpy(),(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            mask_big=torch.tile(self.cfa_256x256,(1,np.where(img_rgbw.shape[2]%self.cfa_256x256.shape[1]==0,img_rgbw.shape[2]//self.cfa_256x256.shape[1],img_rgbw.shape[2]//self.cfa_256x256.shape[1]+1),np.where(img_rgbw.shape[3]%self.cfa_256x256.shape[2]==0,img_rgbw.shape[3]//self.cfa_256x256.shape[2],img_rgbw.shape[3]//self.cfa_256x256.shape[2]+1)))
            # print('mask_big.shape',mask_big.shape) #torch.Size([4, 1536, 2048])
            mask=mask_big[:,:img_rgbw.shape[2],:img_rgbw.shape[3]]
        bayer_rgbw=mask*img_rgbw
        return bayer_rgbw,mask

class LRGBW(ShutterBase):
    def __init__(self,block_size,cfa_size, w_zhi,alpha,test=False, resume=False,init=''):
        super().__init__(test,cfa_size, w_zhi,alpha,resume, block_size,init)
        # print('self.block_size',block_size)
        self.block_size=block_size
        self.cfa_size=cfa_size
        self.init=init
        self.w_zhi=w_zhi
        self.alpha=alpha
        
        # """
        # 柯达 4x4CFA
        rgbw_mask = torch.zeros(size=(4,cfa_size[0],cfa_size[1]),dtype=torch.float32,device=device)#创建RGBW CFA
        # rgbw_mask[:,...]=0.5
        rgbw_mask[0,::4,2::4]=self.w_zhi
        rgbw_mask[0,1::4,3::4]=self.w_zhi
        #green channels
        rgbw_mask[1,::4,::4]=self.w_zhi
        rgbw_mask[1,1::4,1::4]=self.w_zhi
        rgbw_mask[1,2::4,2::4]=self.w_zhi
        rgbw_mask[1,3::4,3::4]=self.w_zhi
        #blue channels
        rgbw_mask[2,2::4,::4]=self.w_zhi
        rgbw_mask[2,3::4,1::4]=self.w_zhi
        #W channel
        rgbw_mask[3,::2,1::2]=self.w_zhi
        rgbw_mask[3,1::2,::2]=self.w_zhi
        # """
        """
        # 1994 4x4CFA
        rgbw_mask = torch.zeros(size=(4,cfa_size[0],cfa_size[1]),dtype=torch.float32,device=device)#创建RGBW CFA
        # R channels
        rgbw_mask[0,0::4,2::4]=self.w_zhi
        rgbw_mask[0,2::4,0::4]=self.w_zhi
        # G channels
        rgbw_mask[1,1::2,1::2]=self.w_zhi
        # B channels
        rgbw_mask[2,0::4,0::4]=self.w_zhi
        rgbw_mask[2,2::4,2::4]=self.w_zhi
        # W channels
        rgbw_mask[3,0::2,1::2]=self.w_zhi
        rgbw_mask[3,1::2,0::2]=self.w_zhi
        # save_image(rgbw_mask[:3], 'cfa_1.png')
        # exit()
        """

        """
        # 2017 8x8CFA
        rgbw_mask = torch.zeros(size=(4,cfa_size[0],cfa_size[1]),dtype=torch.float32,device=device)#创建RGBW CFA
        # R channels
        rgbw_mask[0,1::8,7::8]=self.w_zhi
        rgbw_mask[0,3::8,1::8]=self.w_zhi
        rgbw_mask[0,3::8,3::8]=self.w_zhi
        rgbw_mask[0,5::8,7::8]=self.w_zhi
        rgbw_mask[0,7::8,1::8]=self.w_zhi
        # G channels
        rgbw_mask[1,1::8,3::8]=self.w_zhi
        rgbw_mask[1,3::8,7::8]=self.w_zhi
        rgbw_mask[1,5::8,3::8]=self.w_zhi
        rgbw_mask[1,5::8,5::8]=self.w_zhi
        rgbw_mask[1,7::8,7::8]=self.w_zhi
        # B channel
        rgbw_mask[2,1::8,1::8]=self.w_zhi
        rgbw_mask[2,1::8,5::8]=self.w_zhi
        rgbw_mask[2,3::8,5::8]=self.w_zhi
        rgbw_mask[2,5::8,1::8]=self.w_zhi
        rgbw_mask[2,7::8,3::8]=self.w_zhi
        rgbw_mask[2,7::8,5::8]=self.w_zhi
        # W channel
        rgbw_mask[3,0::2,0::1]=self.w_zhi
        rgbw_mask[3,1::2,0::2]=self.w_zhi
        """
        w=torch.zeros(size=(self.cfa_size[0]*self.cfa_size[1],4,1,1),dtype=torch.float32,device=device) #创建w
        self.mask=torch.zeros(size=(self.cfa_size[0]*self.cfa_size[1],self.block_size[0],self.block_size[1],self.block_size[2]),dtype=torch.float32,device=device) #创建mask
        group_count=0
        for i in range(self.cfa_size[0]):
            for j in range(self.cfa_size[1]):
                w[group_count]=rgbw_mask[:,i,j].reshape(4,1,1) #对w进行初始化
                self.mask[group_count,:,i::self.cfa_size[0],j::self.cfa_size[1]]=1 #对mask进行初始化
                group_count+=1

        # w=torch.normal(0, 0.1, (self.cfa_size[0]*self.cfa_size[1],4,1,1))#产生一个正态分布均值为0，方差为0.1，尺寸为(self.cfa_size[0]*self.cfa_size[1],4,1,1)
        w = F.softmax(w, dim=1)
        # w=torch.rand(size=(self.cfa_size[0]*self.cfa_size[1],4,1,1),device=device) #w 为随机初始化
        self.w=nn.Parameter(w, requires_grad=True)
        '''
        # self.mask_324_487=init_mask(cfa_size=self.cfa_size,c=4,h=324,w=487)
        self.mask_1359_2041=init_mask(cfa_size=self.cfa_size,c=4,h=1359,w=2041)
        # self.mask_2041_1359=init_mask(cfa_size=self.cfa_size,c=4,h=2041,w=1359)
        # self.mask_2193_1460=init_mask(cfa_size=self.cfa_size,c=4,h=2193,w=1460)
        self.mask_1460_2193=init_mask(cfa_size=self.cfa_size,c=4,h=1460,w=2193)
        '''
    def forward(self, image, train=False):
        cfa_image=torch.zeros(size=(image.shape),dtype=torch.float32,device=device)
        cfa=torch.zeros(size=(image.shape[1:]),dtype=torch.float32,device=device)
        if train:
            # cfa_image, cfa = self.Proxy_function(cfa_image=cfa_image, cfa=cfa, init=self.init, image=image, w=self.w,
                                            # mask=self.mask)
            cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=self.mask,alpha=self.alpha)
        else:
            # '''
            mask=torch.zeros(size=(self.cfa_size[0]*self.cfa_size[1],image.shape[1],image.shape[2],image.shape[3]),device=device)
            group_count=0
            for i in range(self.cfa_size[0]):
                for j in range(self.cfa_size[1]):
                    mask[group_count,:,i::self.cfa_size[0],j::self.cfa_size[1]]=1
                    group_count+=1
            # cfa_image, cfa = self.Proxy_function(cfa_image=cfa_image, cfa=cfa, init=self.init, image=image, w=self.w,
                                        #    mask=mask)
            cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=mask,alpha=self.alpha)
            # '''
            """
            # print('image.shape[1:4]',image.shape[1:4])
            # print('self.mask_324_487.shape[1:4]',self.mask_324_487.shape[1:4])
            # if image.shape[1:4]==self.mask_324_487.shape[1:4]:
                # print('self.mask_324_487.shape',self.mask_324_487.shape)
                # cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=self.mask_324_487)
            if image.shape[1:4]==self.mask_1359_2041.shape[1:4]:
                print('11111111111')
                cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=self.mask_1359_2041)
            # elif image.shape[1:4]==self.mask_2041_1359.shape[1:4]:
            #     cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=self.mask_2041_1359)
            # elif image.shape[1:4]==self.mask_2193_1460.shape[1:4]:
            #     cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=self.mask_2193_1460)
            elif image.shape[1:4]==self.mask_1460_2193.shape[1:4]:
                cfa_image,cfa=Proxy_function(cfa_image=cfa_image,cfa=cfa,init=self.init,image=image,w=self.w,mask=self.mask_1460_2193)
            else:
                print("未进行合适尺寸mask的初始化,退出程序")
                sys.exit(1) #退出程序函数
        """
        return cfa_image,cfa
        # return image,cfa


def init_mask(cfa_size,c,w,h):
    #mask的初始化
    mask=torch.zeros(size=(cfa_size[0]*cfa_size[1],c,h,w),device=device)
    group_count=0
    for i in range(cfa_size[0]):
        for j in range(cfa_size[1]):
            mask[group_count,:,i::cfa_size[0],j::cfa_size[1]]=1
            group_count+=1
    return mask


