import torch.nn as nn
import shutters.myshutters as myshutters
import torch
from nets.unet import UNet
from nets.rlfn import RLFN
from nets.NAFNet_arch import NAFNet
from nets.network_unet import SIMDUNet
from nets.rrdb import RRDBNet
from nets.mysgnet1 import MySgNet1
from nets.mysgnet import MySGNet
from nets.mprnet import MPRNet_s2
from nets.rdunet import RDUNet
from nets.transform_modules import TileInterp    # TreeMultiRandom
from src.myTreeMultiRandom import MyTreeMultiRandom
from src.myTreeMultiRandom import MyTreeScatter
from nets.dncnn import DnCNN, init_weights
from PConv.PConv import PartialConv2d,PartialConv2d_Not_Official
from nets.unet_myself import UNetMyself
from torchvision.utils import save_image
import torch.nn.functional as F
from nets.sgnet import NET
from nets.shufflemixer_arch import ShuffleMixer
from nets.msanet import MSANet
from nets.network_unet import UNetRes

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def define_myshutter(shutter_type,args,test=False):
    return myshutters.Shutter(block_size=args.block_size,shutter_type=shutter_type,cfa_size=args.cfa_size,
                w_zhi=args.w_zhi,alpha=args.alpha,test=test,init=args.init)

def define_mymodel(shutter, decoder, args, get_coded=False):
    #在编码器和解码器之间定义任何特殊的插值模块
    if args.interp=='pconv' and (args.shutter=='rgbw'):
        print('*******添加RGBW-PConvModel*********')
        return PConvModel(shutter,decoder,in_channels=4,out_channels=4)
    elif args.interp=='scatter' and args.shutter=='rgbw':
        print("******添加TreeRGBWModel******")
        return TreeRGBWModel(shutter,
                         decoder,
                         get_coded=get_coded,
                         k=args.k,
                         p=args.p,
                         num_channels=4,
                         )#原版sz=args.block_size[-1],
    elif args.interp=='scatter'and args.shutter=='rgb':
        print("******添加TreeRGBModel******")
        return TreeRGBWModel(shutter,
                         decoder,
                         get_coded=get_coded,
                         k=args.k,
                         p=args.p,
                         num_channels=3,
                         )#原版sz=args.block_size[-1],
    elif args.interp=='pconv' and args.shutter=='rgb':
        print('*******添加RGB-PConvModel*********')
        return PConvModel(shutter,decoder,in_channels=3,out_channels=3)
    elif args.interp=='scatter' and (args.shutter=='lrgbw'): #args.interp=='scatter' and args.shutter=='lsvpe':
        print("******添加TreeRGBWModel******")
        return TreeRGBWModel(shutter,
                         decoder,
                         get_coded=get_coded,
                         k=args.k,
                         p=args.p,
                         num_channels=4,
                         )#原版sz=args.block_size[-1],
    elif args.interp=='pconv' and args.shutter=='lrgbw' or args.shutter=='nipsrgbw':
        print('*******添加LRGBW-PConvModel*********')
        return PConvModel(shutter,decoder,in_channels=4,out_channels=4)
    elif args.interp=='gaussian' and args.shutter=='lrgbw':
        print('*******添加LRGBW-Gaussian插值*********')
        return GaussianModel(shutter=shutter,decoder=decoder,gaussian_weight_size=args.gaussian_weight_size)

    elif args.interp=='gaussian' and args.shutter=='rgbw' or args.shutter=='nipsrgbw' or args.shutter=='luorgbw'  or  args.shutter=='ourrgbw' or  args.shutter=='wangrgbw' or args.shutter=='cfzrgbw' or args.shutter=='kaizurgbw' or args.shutter=='gindelergbw' or args.shutter=='binningrgbw'or args.shutter=='sonyrgbw' or args.shutter=='yamagamirgbw' or args.shutter=='hondargbw' or args.shutter=='randomrgbw'  or args.shutter=='hamiltonrgbw':
        print('*******添加RGBW-Gaussian插值*********')
        return GaussianModel(shutter=shutter,decoder=decoder,gaussian_weight_size=args.gaussian_weight_size)
    elif args.interp=='bilinear' and args.shutter=='bayerrgb':
        print('*******添加BayerRGB-双线性插值*********')
        return BayerBilinearModel(shutter=shutter,decoder=decoder)
        
    
    elif args.interp=='none' and args.shutter=='rgbw':
        print('*****不添加插值模块*******')
        return Model(shutter=shutter,decoder=decoder,get_coded=get_coded)
    
    raise NotImplementedError('Interp + Shutter combo has not been implemented')

def define_mydecoder(model_name, args):
    if args.decoder == 'none':
        return None
    elif args.shutter=='rgbw' and args.interp=='scatter':
        in_ch=4
        out_ch=4
    elif args.shutter=='rgbw' and args.interp=='pconv':
        in_ch=4
        out_ch=4
    elif args.shutter=='rgbw' and args.interp=='gaussian' or args.shutter=='nipsrgbw' or  args.shutter=='ourrgbw' or  args.shutter=='wangrgbw' or args.shutter=='luorgbw'or args.shutter=='cfzrgbw' or args.shutter=='kaizurgbw' or args.shutter=='gindelergbw' or args.shutter=='binningrgbw'or args.shutter=='sonyrgbw' or args.shutter=='yamagamirgbw' or args.shutter=='hondargbw' or args.shutter=='randomrgbw' or args.shutter=='hamiltonrgbw':
        in_ch = 4
        out_ch = 3
    elif args.shutter=='lrgbw' and args.interp=='gaussian':
        in_ch = 4
        out_ch = 3
    elif args.shutter == 'bayerrgb' and args.interp == 'scatter':
        in_ch = 3
        out_ch = 3
    elif args.shutter=='bayerrgb' and args.interp=='bilinear':
        in_ch=3
        out_ch=3
    elif args.shutter == 'rgb' and args.interp == 'pconv':
        in_ch = 3
        out_ch = 3
    elif args.shutter=='lrgbw'  and args.interp=='scatter': #elif args.shutter=='lsvpe' and args.interp=='scatter':
        in_ch=4
        out_ch=4
    elif args.shutter=='lrgbw'  and args.interp=='pconv': #elif args.shutter=='lsvpe' and args.interp=='pconv':
        in_ch=4
        out_ch=4
    elif args.shutter=='rgbw' and args.interp=='none':
        in_ch=4
        out_ch=4
    else:
        raise NotImplementedError
    if model_name=='myunet':
        return UNetMyself(n_channels=in_ch,n_classes=out_ch)
    elif model_name == 'unet':
        #对于RGBW必须要加BatchNormal层，否则没有效果
        return UNet(in_ch=in_ch, out_ch=out_ch, depth=6, wf=5, padding=True, batch_norm=False, up_mode='upconv') #upsample 原版up_mode='upconv' batch_norm=False
    elif model_name=='sgnet':
        return NET(sr_n_resblocks=6,dm_n_resblocks=6,channels=24,scale=2,denoise=False,
                block_type='rrdb',act_type='relu',bias=False,norm_type=None)
    elif model_name=='mysgnet':
        return MySGNet(in_nc=3, out_nc=3, 
                 nc=[64, 128, 256, 512], nb=2, act_mode='R', 
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='simdunet':
        return SIMDUNet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=3, act_mode='R', 
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='rlfn':
        return RLFN(in_channels=4,out_channels=3,feature_channels=52)
    elif model_name=='nafnet':
        middle_blk_num=24
        enc_blks = [2, 2, 4, 8]
        dec_blks = [2, 2, 2, 2]
        return NAFNet(img_channel=4, width=32, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks) 
    elif model_name=='mysgnet1':
        return MySgNet1(inchannels=4,outchannels=3,
                 kernal_size=3)
    elif model_name=='rrdbnet':
        return RRDBNet(channels=64,act_type='leakyrelu',bias=True,norm_type=None)
    elif model_name=='rdunet':
        return RDUNet(inchannels=4,outchannels=3,base_filters=64) #原版base_filters=64，可学习测试显存爆炸
    elif model_name=='shufflemixer':
        return ShuffleMixer(n_feats=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscaling_factor=1)
    elif model_name =='msanet':
        return MSANet(input_channel=4, output_channel=3)
    elif model_name=='unetres':
        return UNetRes(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=2, act_mode='R', 
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    raise NotImplementedError('Model not specified correctly')

class PConvModel(nn.Module):
    def __init__(self, shutter, decoder,in_channels,out_channels):
        super(PConvModel, self).__init__()
        self.shutter = shutter
        self.decoder = decoder
        # print('self.shutter',self.shutter) # RGBW()
        # print('self.decoder',self.decoder) #UNet(.....


        #定义卷积层
        # self.conv=PartialConv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False, multi_channel = True,return_mask=True).to(device) #官方
        self.pconv = PartialConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                  multi_channel=True, return_mask=True).to(device)  # 官方
        self.pconv1 = PartialConv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False,
                                  multi_channel=False, return_mask=True).to(device)  # 官方
        self.pconv_not_guanfang=PartialConv2d_Not_Official(in_channels=in_channels,out_channels=out_channels,kernel_size=3,
                                stride=1,padding=1,bias=False) #非官方
        self.pconv_not_guanfang1=PartialConv2d_Not_Official(in_channels=1,out_channels=1,kernel_size=7,stride=1,padding=3,bias=False) #非官方
                                
        # 定义unet层
        # self.unet=UNet(n_channels=3,n_classes=3).to(device)
    # def forward(self,image): #原来版本
    def forward(self, image,train=True):#修改版本
        # rgbw_raw,mask=self.shutter(image) #原来版本
        rgbw_raw, mask = self.shutter(image,train=train)#修改版本
        # print('rgbw_raw.shape',rgbw_raw.shape) #torch.Size([1, 4, 512, 512])
        # print('mask.shape',mask.shape) #torch.Size([4, 324, 487])
        # print('mask',mask)
        mask_4=mask[None,:,:,:]
        # print('mask.shape',mask.shape) #torch.Size([1, 4, 324, 487])
        # rgbw_raw.register_hook(print)  #为0
        # mask_4.register_hook(print)  # 为0
        # rgbw,update_mask=self.pconv(rgbw_raw,mask_4) #官方卷积、提供mask
        
        # print(rgbw==0)
        # rgbw, update_mask = self.conv(rgbw_raw)#不提供mask
        # print('rgbw.shape',rgbw.shape) #torch.Size([1, 4, 512, 512])
        # print(update_mask==0)
        # rgbw.register_hook(print) #不为0
        # rgbw,update_mask=self.pconv_not_guanfang(rgbw_raw,mask_4) #非官方卷积
        
        
        
        r_raw=rgbw_raw[:,0,:,:][:,None,:,:]
        g_raw=rgbw_raw[:,1,:,:][:,None,:,:]
        b_raw=rgbw_raw[:,2,:,:][:,None,:,:]
        w_raw=rgbw_raw[:,3,:,:][:,None,:,:]
        r_mask=mask[0,:,:][None,None,:,:]
        g_mask=mask[1,:,:][None,None,:,:]
        b_mask=mask[2,:,:][None,None,:,:]
        w_mask=mask[3,:,:][None,None,:,:]
        # print('r_raw.shape',r_raw.shape) #torch.Size([4, 1, 256, 256])
        # print('r_mask.shape',r_mask.shape) #torch.Size([1, 1, 256, 256])
        '''
        #四个通道单独处理  官方代码
        r_img,update_mask_r=self.pconv1(r_raw,r_mask)
        g_img,update_mask_g=self.pconv1(g_raw,g_mask)
        b_img,update_mask_b=self.pconv1(b_raw,b_mask)
        w_img,update_mask_w=self.pconv1(w_raw,w_mask)
        # print('r_img.shape',r_img.shape) #torch.Size([4, 1, 256, 256])
        # print(r_img==0)
        # print(g_img==0)
        # print(b_img==0)
        # print(w_img==0)
        rgbw=torch.cat((r_img,g_img,b_img,w_img),dim=1)
        # print('rgbw.shape',rgbw.shape) #torch.Size([4, 4, 256, 256])
        # print(rgbw==0) 
        # print('rgbw1',rgbw1)
        '''

        #四个通道单独处理  非官方代码
        r_img,update_mask_r=self.pconv_not_guanfang1(r_raw,r_mask)
        g_img,update_mask_g=self.pconv_not_guanfang1(g_raw,g_mask)
        b_img,update_mask_b=self.pconv_not_guanfang1(b_raw,b_mask)
        w_img,update_mask_w=self.pconv_not_guanfang1(w_raw,w_mask)
        rgbw=torch.cat((r_img,g_img,b_img,w_img),dim=1)
        # exit()
        output=self.decoder(rgbw)
        # print('output.shape',output.shape) #torch.Size([1, 1, 512, 512])
        # output=self.unet(image_output)
        return output,mask[:3]


class TreeRGBWModel(nn.Module):
    def __init__(self, shutter, decoder, get_coded=False,
                 k=3, p=1, num_channels=8):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        # print('self.shutter',self.shutter) # LSVPE() Quad()
        self.decoder = decoder
        # self.shutter_name = shutter_name
        # self.tree=MyTreeMultiRandom(k=k,p=p,num_channels=num_channels) #原来版本
        self.tree=MyTreeScatter(k=k,p=p,num_channels=num_channels)
        # self.tree = TreeMultiRandom(sz=sz, k=k, p=p, num_channels=num_channels)

    # def forward(self, input): #原来版本
    def forward(self, input,train=True):  # 修改版本
        # print('input.shape',input.shape) #torch.Size([1, 4, 186, 317])     RGBW:torch.Size([1, 4, 512, 512])  torch.Size([2, 8, 512, 512])   Quad torch.Size([2, 8, 512, 512])
        # print('self.shutter',self.shutter)
        # coded,mask = self.shutter(input) #原来版本
        coded, cfa = self.shutter(input,train=train)  # 修改版本
        # coded.register_hook(print)
        # print('coded111111111',coded)
        # print('coded',coded)
        # print(coded==0)
        # save_image(coded[:, :3, :, :], '../image_result/cfa_img.png')
        # print('coded.shape',coded.shape) #RGBW torch.Size([1, 4, 512, 512])  torch.Size([2, 1, 512, 512])   Quad torch.Size([2, 1, 512, 512])   RGBWtorch.Size([1, 4, 512, 512])
        # coded.register_hook(print)
        multi = self.tree(coded) #原来版本
        # coded.register_hook(print)
        # print('code',coded)
        # multi = self.tree(input) #新版本
        # print('multi',multi)
        # print(multi==0)

        #保存scatter之后的图片
        # save_image(multi[:,:3,:,:],'../image_result/scatter_img.png')
        
        # print('multi.shape',multi.shape) #RGBW torch.Size([1, 4, 8, 8])torch.Size([2, 9, 512, 512])   Quad torch.Size([2, 4, 512, 512])
        # multi.register_hook(print)
        x = self.decoder(multi)
        # coded.register_hook(print)
        # x.register_hook(print)
        # x=self.decoder(coded)
        # print('x.shape',x.shape)#RGBW torch.Size([1, 4, 512, 512])  torch.Size([2, 1, 512, 512])            Quad torch.Size([2, 1, 512, 512])
        if self.get_coded:
            return x, coded
        return x,cfa[:3]
        # return coded,mask

class Model(nn.Module):
    def __init__(self, shutter, decoder,  get_coded=False):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder

    def forward(self, input, train=True):
        # print('self.shutter',self.shutter)
        # print('input.shape',input.shape)#torch.Size([1, 4, 150, 150])
        coded,mask= self.shutter(input, train=train)
        # print('coded.shape',coded.shape)
        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x

class GaussianModel(nn.Module):
    def __init__(self, shutter, decoder,gaussian_weight_size):
        super(GaussianModel, self).__init__()
        self.shutter = shutter
        self.decoder = decoder
        self.Gaussian_weight_size=gaussian_weight_size
        # print(self.Gaussian_weight_size)
        # exit()
        # self.Gaussian_weight=torch.normal(mean=0,std=9/6,size=(4,1,9,9),device=device)
        weight=torch.normal(mean=0,std=(self.Gaussian_weight_size/(self.Gaussian_weight_size-2)),size=(self.Gaussian_weight_size,self.Gaussian_weight_size),device=device)
        # print('weight',weight)
        # exit()
        weight=weight[None,None,:,:]
        self.Gaussian_weight=torch.cat((weight,weight,weight,weight),dim=0)

    def forward(self, image,train=True):
        # print('self.Gaussian_weight',self.Gaussian_weight)
        rgbw_raw, mask = self.shutter(image,train=train)
        # save_image(rgbw_raw[:,:3,:,:],'../image_result/cfa_img.png')
        mask_4=mask[None,:,:,:]
        # print('rgbw_raw.shape',rgbw_raw.shape)
        # print('mask_4.shape',mask_4.shape) #torch.Size([1, 4, 256, 256])
        # print('rgbw_raw',rgbw_raw)
        rgbw_Gaussian=F.conv2d(input=rgbw_raw,weight=self.Gaussian_weight,stride=1,
                                padding=self.Gaussian_weight_size//2,groups=4,bias=None)
        mask_Gaussian=F.conv2d(input=mask_4,weight=self.Gaussian_weight,stride=1,
                                padding=self.Gaussian_weight_size//2,groups=4,bias=None)
        epsilon=0.1/255
        rgbw=rgbw_Gaussian/(mask_Gaussian+epsilon)
        # 把原来的像素值赋值回去
        # print(rgbw_raw)
        # index = torch.where(rgbw_raw != 0,1,2) #torch1.1.0版本
        # index=torch.nonzero(rgbw_raw)
        # print(index)
        # print(len(index))
        index = torch.where(rgbw_raw != 0)
        # print(index)
        # exit()
        data = rgbw_raw
        temp = data[index]
        rgbw[index] = temp
        # print(rgbw == rgbw_raw)
        # exit()
        # print(rgbw_Gaussian==0)
        # print(mask_Gaussian==0)
        # print(rgbw==0)
        # print('rgbw',rgbw)
        # exit()
        # print(rgbw==0)
        # save_image(rgbw[:,:3,:,:],'../image_result/gaussian_img.png')
        # exit()
        output=self.decoder(rgbw)

        # output=self.decoder(rgbw_Gaussian)
        # output.register_hook(print)
        return output,mask[:3]
    
class BayerBilinearModel(nn.Module):
    def __init__(self, shutter, decoder):
        super(BayerBilinearModel, self).__init__()
        self.shutter = shutter
        self.decoder = decoder
        r""" Initialize with bilinear interpolation
        输入为4维，batch_size channels height weight
        """
        F_r = torch.FloatTensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
        F_b = F_r
        F_g = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
        bilinear_filter = torch.stack([F_r, F_g, F_b])[:, None]
        self.bilinear_filter = bilinear_filter.to(device)

    def forward(self, rgb_image,train=True):
        # print('self.Gaussian_weight',self.Gaussian_weight)
        # rgb_image=image[:,:3,:,:]
        rgb_raw, mask = self.shutter(rgb_image,train=train)
        # save_image(rgb_raw,'rgbraw.png')
        rgb=F.conv2d(rgb_raw, self.bilinear_filter, padding=1, groups=3)
        # save_image(rgb,'rgb.png')
        # exit(0)
        output=self.decoder(rgb)

        # output=self.decoder(rgbw_Gaussian)
        # output.register_hook(print)
        return output,mask[:3]
