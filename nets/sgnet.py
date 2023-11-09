import nets.common as common
import torch.nn as nn
#from model.common import SFTLayer
import pdb
import torch
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pdb
import cv2
from nets.unet_myself import UNetMyself
from nets.network_unet import SIMDUNet
from nets.pac import PacConvTranspose2d,PacConv2d
"""
given LR bayer  -> output HR noise-free RGB
"""
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class m_res(nn.Module):  #去马赛克相关模型
    # def __init__(self, opt):
    def __init__(self,sr_n_resblocks,dm_n_resblocks,channels,scale,denoise,
                block_type,act_type,bias,norm_type):
        super(m_res, self).__init__()
        '''
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type 
        '''
        sr_n_resblocks = sr_n_resblocks
        dm_n_resblocks = dm_n_resblocks
        sr_n_feats = channels
        dm_n_feats = channels
        scale = scale

        denoise = denoise
        block_type = block_type
        act_type = act_type
        bias = bias
        norm_type = norm_type

        self.r1 = common.RRDB2(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2) #原版
        # self.r1 = common.RRDB2(nc=dm_n_feats,gc=32,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        self.r2 = common.RRDB2(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.r3 = common.RRDB2(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.r4 = common.RRDB2(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.r5 = common.RRDB2(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.r6 = common.RRDB2(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.final = common.ConvBlock(in_channelss=dm_n_feats,out_channels=dm_n_feats,kernel_size=3, bias=bias)#原版
        # self.final = common.ConvBlock(in_channelss=dm_n_feats,out_channels=3,kernel_size=3, bias=bias)
    def forward(self, x):
        output = self.r1(x)
        output = self.r2(output)
        # output = self.r3(output)
        # output = self.r4(output) #原版
        # output = self.r5(output) #原版
        # output = self.r6(output) #原版
        # print('output.shape', output.shape) # torch.Size([1, 64, 256, 256])
        # exit()
        # return self.final(output)
        return output



class green_res(nn.Module):
    # def __init__(self, opt):
    def __init__(self,sr_n_resblocks,dm_n_resblocks,channels,scale,denoise,
                block_type,act_type,bias,norm_type):
        super(green_res, self).__init__()
        '''
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type 
        '''
        sr_n_resblocks = sr_n_resblocks
        dm_n_resblocks = dm_n_resblocks
        sr_n_feats = channels
        dm_n_feats = channels
        scale = scale

        denoise = denoise
        block_type = block_type
        act_type = act_type
        bias = bias
        norm_type = norm_type
        # self.head = common.ConvBlock(in_channelss=2 ,out_channels=dm_n_feats,kernel_size=5, act_type=act_type, bias=True)#原版
        # self.head = common.ConvBlock(in_channelss=1 ,out_channels=dm_n_feats,kernel_size=5, act_type=act_type, bias=True)
        self.r1 = common.RRDB(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2) #原版
        # self.r1 = common.RRDB2(nc=dm_n_feats,gc=32,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.r2 = common.RRDB(nc=dm_n_feats,gc=dm_n_feats,kernel_size=3,stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
        # self.r3 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        # self.r4 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        # self.r5 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        # self.r6 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        # self.final = common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=bias) #原版
        # self.final = common.ConvBlock(in_channelss= dm_n_feats, out_channels= 1,kernel_size= 3, bias=bias)
        '''
        self.up = nn.Sequential(
           common.Upsampler(scale= 2,n_feats= dm_n_feats,norm_type= norm_type,act_type= act_type, bias=bias),
           common.ConvBlock(in_channelss= dm_n_feats,out_channels= 1 ,kernel_size= 3, bias=True),
           nn.LeakyReLU(0.2, inplace = True)
        )
        '''
    def forward(self, x):
        # print('xxxx.shape',x.shape)
        output = self.r1(x)
        # output = self.r1(self.head(x)) #原版
        # output = self.r2(output)
        # output = self.final(output) +self.head(x) #原版
        # output = self.up(output) #原版 上采样
        #output = self.r3(output)
        #output = self.r4(output)
        #output = self.r5(output)
        #output = self.r6(output)
        return output #self.final(output)
        # return self.final(output)
   
class NET(nn.Module):
    # def __init__(self, opt):
    def __init__(self,sr_n_resblocks,dm_n_resblocks,channels,scale,denoise,
                block_type,act_type,bias,norm_type):
        super(NET, self).__init__()
        '''
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type
        '''
        sr_n_resblocks = sr_n_resblocks
        dm_n_resblocks = dm_n_resblocks
        sr_n_feats = channels
        dm_n_feats = channels
        scale = scale

        denoise = denoise
        block_type = block_type
        act_type = act_type
        bias = bias
        norm_type = norm_type
        
        '''
        # define sr module 定义超分辨率模型
        if denoise:
            m_sr_head = [common.ConvBlock(in_channelss=6, out_channels=sr_n_feats, kernel_size=5,
                                          act_type=act_type, bias=True)]
        else:
            m_sr_head = [common.ConvBlock(in_channelss=4, out_channels=sr_n_feats, kernel_size=5,
                                          act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_sr_resblock = [common.RRDB(nc=sr_n_feats,gc=sr_n_feats,kernel_size=3,
                                         stride=1,bias=bias,norm_type=norm_type,act_type=act_type,res_scale=0.2)
                             for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_sr_resblock = [common.DUDB(sr_n_feats, 3, 1, bias,
                                         norm_type, act_type, 0.2)
                             for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'res':
            m_sr_resblock = [common.ResBlock(sr_n_feats, 3, norm_type,
                                             act_type, res_scale=1, bias=bias)
                             for _ in range(sr_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        '''
        
        '''
        m_sr_resblock += [common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=bias)]
        m_sr_up = [common.Upsampler(scale, sr_n_feats, norm_type, act_type, bias=bias),
                   common.ConvBlock(sr_n_feats, 4, 3, bias=True)]

        # branch for sr_raw output
        m_sr_tail = [nn.PixelShuffle(2)]
        '''

        # define demosaick module  定义去马赛克模型
        m_dm_head = [common.ConvBlock(in_channelss=3,out_channels=dm_n_feats,kernel_size=5,
                                      act_type=act_type, bias=bias)]
        
        w_dm_head = [common.ConvBlock(in_channelss=1,out_channels=dm_n_feats,kernel_size=5,
                                      act_type=act_type, bias=bias)]

        if block_type.lower() == 'rrdb':
            # m_dm_resblock = m_res(opt) #[common.RRDB(dm_n_feats, dm_n_feats, 3,
                                         #1, bias, norm_type, act_type, 0.2)
                             #for _ in range(dm_n_resblocks)]
            m_dm_resblock=m_res(sr_n_resblocks, dm_n_resblocks, channels, 
                                scale, denoise, block_type, act_type, bias, norm_type)
        elif block_type.lower() == 'dudb':
            m_dm_resblock = [common.DUDB(dm_n_feats, 3, 1, bias,
                                         norm_type, act_type, 0.2)
                             for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'res':
            m_dm_resblock = [common.ResBlock(dm_n_feats, 3, norm_type,
                                             act_type, res_scale=1, bias=bias)
                             for _ in range(dm_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        #m_dm_resblock += [common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=bias)]
       
        # m_dm_up = [common.Upsampler(scale= 2,n_feats= dm_n_feats,norm_type= norm_type,act_type= act_type, bias=bias)] #原版
                   #common.ConvBlock(dm_n_feats, 3, 3, bias=True)]

        # self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.Sequential(*m_sr_resblock)),
                                    #   *m_sr_up) #原版
        '''
        self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.Sequential(*m_sr_resblock)),
                                      ) #非原版，去掉上采样模块
        self.sr_output = nn.Sequential(*m_sr_tail)
        '''
        self.w_dm1 = nn.Sequential(*w_dm_head)
        self.model_dm1 = nn.Sequential(*m_dm_head)
        self.model_dm2 = m_dm_resblock
        self.model_final = common.ConvBlock(in_channelss=dm_n_feats,out_channels=3,kernel_size=3, bias=bias)
        self.w_final = common.ConvBlock(in_channelss=dm_n_feats,out_channels=1,kernel_size=3, bias=bias)
        # self.model_dm3 = nn.Sequential(*m_dm_up) #原版


        # greenresblock = green_res(opt)
        # w通道，输入输出均为1
        # channels=1
        greenresblock = green_res(sr_n_resblocks, dm_n_resblocks, channels, 
                                scale, denoise, block_type, act_type, bias, norm_type)
        self.green = greenresblock
        # self.green_myunet= UNetMyself(n_channels=1,n_classes=1) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
        '''
        #self.sft = SFTLayer()
        self.combine = nn.Sequential(
                common.ConvBlock(in_channelss= dm_n_feats+1,out_channels= dm_n_feats,kernel_size= 1 , bias=True),
                # nn.LeakyReLU(0.2 , inplace = True) #原版
                nn.LeakyReLU(0.2 , inplace = False)
        )
        '''
        
        self.greenup = nn.Sequential(
                common.ConvBlock(in_channelss= 1,out_channels= 4,kernel_size= 1 , bias=True),
                # nn.LeakyReLU(0.2 , inplace = True), #原版
                nn.LeakyReLU(0.2 , inplace = False),
                # nn.ReLU(inplace=True),
                
                common.ConvBlock(in_channelss= 4,out_channels= 8,kernel_size= 1 , bias=True),
                # nn.LeakyReLU(0.2 , inplace = True) #原版
                nn.LeakyReLU(0.2 , inplace = False)

        )

        # PacConvTranspose2d()带上采样的卷积
        # self.pac = PacConvTranspose2d(in_channels= 64,out_channels= 64,kernel_size=5, stride=2, padding=2, output_padding=1)
        #
        # self.pac = PacConv2d(in_channels=64, out_channels=64, kernel_size=5,stride=1,padding=2)  # 
        self.pac = PacConv2d(in_channels=dm_n_feats, out_channels=dm_n_feats, kernel_size=5,stride=1,padding=2)
        self.final = common.ConvBlock(in_channelss= 72 ,out_channels= 3,kernel_size= 3 , bias=True)
        self.final_myunet= UNetMyself(n_channels=4,n_classes=3)
        self.final_drunet=DResUNet(in_nc=64,out_nc=3,nc=[64, 128, 256, 512], nb=2, act_mode='R', 
                 downsample_mode="strideconv", upsample_mode="convtranspose")
        self.norm = nn.InstanceNorm2d(1)
        
    def density(self , x):
        x = torch.clamp(x**(1/2.2)*255,0.,255.).detach()
        b,w,h = x.shape
        
        im= np.array(x[0].cpu()).astype(np.uint8)
        im = Image.fromarray(im)
        im_blur = im.filter(ImageFilter.GaussianBlur(radius=3))
        im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
        im_minus = np.uint8(im_minus)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        im_sum = torch.from_numpy(cv2.dilate(im_minus, kernel).astype(np.float))
        im_sum = im_sum.unsqueeze(0)
        #print(im_sum.shape)    
        for i in range(1,b):
            im= np.array(x[i].cpu()).astype(np.uint8)
            im = Image.fromarray(im)
            im_blur = im.filter(ImageFilter.GaussianBlur(radius=5))
            im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
            im_minus = np.uint8(im_minus)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
            im_minus = cv2.dilate(im_minus, kernel).astype(np.float)
            im_minus = torch.from_numpy(im_minus).unsqueeze(0)
            #print(im_minus.shape)
            im_sum = torch.cat([im_sum,im_minus] , 0 )
        # return im_sum.unsqueeze(1).float().cuda(3)
        return im_sum.unsqueeze(1).float().to(device)
        
    def forward(self, x ):
        # estimate density map 估计密度图
        # print('gt_image.shape',gt_image.shape) #torch.Size([8, 4, 128, 128])
        # dmap = x.clone() #原版
        # dmap = (dmap[:,0,:,:] + (dmap[:,1,:,:]+dmap[:,2,:,:])/2 + dmap[:,3,:,:])/3 #原版
        """
        dmap = gt_image.clone() 
        dmap = (dmap[:,0,:,:]+dmap[:,1,:,:]+dmap[:,2,:,:]+dmap[:,3,:,:])/4
        dmap = self.density(dmap)
        # print('dmap.shape', dmap.shape) #torch.Size([8, 1, 128, 128]) 
        dmap = self.norm(dmap).detach()
        # print('dmap.shape', dmap.shape) #torch.Size([8, 1, 128, 128]) 
        # exit()
        """
        
        '''
        # super resoliton in the task, JDDS
        # x = self.model_sr(torch.cat([x,dmap],1))#原版    

        print('x.shape',x.shape) #torch.Size([1, 4, 256, 256])  
        x=self.model_sr(x) 
        print('x.shape',x.shape)#torch.Size([1, 4, 256, 256])  
        x1 = x
        sr_raw = self.sr_output(x)
        print('sr_raw.shape',sr_raw.shape)
        '''

        # demosaic and denoise
        #加入密度图
        # x=torch.cat([x,dmap],dim=1) 
        # print('x.shape',x.shape) #torch.Size([8, 5, 128, 128])
        # exit()
        # x1=x
        x_rgb=x[:,0:3,:,:]
        x_w=x[:,3:4,:,:]
        # print(x_w.shape)
        # x2 = self.model_dm1(x[:,0:3,:,:])
        x1_rgb = self.model_dm1(x_rgb)
        x1_w = self.w_dm1(x_w)
        # print('x.shape',x2.shape) #torch.Size([1, 64, 256, 256])
        # green_output = self.green(x1[:,1:3,:,:].detach()) #原版
        # print(x1[:,3:4,:,:].shape) #torch.Size([1, 1, 256, 256])
        # exit()
        # green_output = self.green(x1[:,3:4,:,:].detach()) 
        # green_output = self.green(x1[:,3:4,:,:])
        # green_output = self.green(x1_w)  
        
        # green_output=self.green_myunet(x1[:,3:4,:,:].detach(),1) #绿色通道接入unet

        # print('green_output.shape',green_output.shape) #torch.Size([1, 1, 256, 256])
        # print('self.model_dm2(x).shape',self.model_dm2(x).shape) #torch.Size([1, 64, 256, 256])
        x2_rgb =  x1_rgb + self.model_dm2(x1_rgb)
        x2_w = x1_w + self.green(x1_w)
        # x3_rgb = self.model_final(x2_rgb)
        # x3_w = self.w_final(x2_w)
        # print('x.shape',x.shape) #x.shape x.shape torch.Size([1, 64, 256, 256])
       
        # g_combine = self.greenup(green_output)
        # print('g_combine.shape',g_combine.shape) #torch.Size([1, 8, 256, 256])
        # combine two branch using adaptive conv  采用自适应变换组合两个分支
        # x = self.pac(x,g_combine) #原版pac为带上采样pac，此处为普通pac，非上采样pac
        # print('x.shape', x.shape) #torch.Size([1, 64, 256, 256])
        # x = self.final(x) #一个卷积
        # x=torch.cat((x,green_output),dim=1)
        # x4_w = self.greenup(x3_w)
        # x = torch.cat((x3_rgb,x3_w),dim=1)
        x = torch.cat((x2_rgb,x2_w),dim=1)
        x=self.DResUNet(x)
        # x = self.final_myunet(x,1)
        # print('x.shape', x.shape) #torch.Size([1, 3, 256, 256])
        # x = self.final(x)
        # return sr_raw, x , green_output #原版
        return x


