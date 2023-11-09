import nets.common as common
from nets.unet_myself import UNetMyself
from nets.network_unet import UNetRes,SIMDUNet
from nets.rdunet import RDUNet
import torch.nn as nn
import torch
# from nets.pac import PacConvTranspose2d,PacConv2d
from nets.msanet import AdaFeatBlock


class MySgNet1(nn.Module):
    def __init__(self,inchannels,outchannels,
                 kernal_size,norm_type=False, act_type='prelu',
                 bias=False, res_scale=1):
        super(MySgNet1, self).__init__()
        if act_type=='relu':
            self.act_mode='R'
        elif act_type=='leakyrelu':
            self.act_mode='L'
        elif act_type=="prelu":
            self.act_mode='P'
        self.rgb_head = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1)
        self.w_head = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
        self.rgb_res_block=common.ResBlock(n_feats=3,kernel_size=3,
            norm_type=False, act_type=act_type, bias=False, res_scale=1)
        self.w_res_block=common.ResBlock(n_feats=1,kernel_size=3,
            norm_type=False, act_type=act_type, bias=False, res_scale=1)
        
        # self.rgb_AFeB=AdaFeatBlock(in_channel=3, out_channel=3)
        # self.w_AFeB=AdaFeatBlock(in_channel=1, out_channel=1)
        # self.unetmyself=UNetMyself(n_channels=3,n_classes=3)
        #RLFB+UNet
        # self.rlfbunet=RLFBUNet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=1, act_mode='R', 
                #  downsample_mode="strideconv", upsample_mode="convtranspose")
        
        #SELFB+UNet
        # self.srlfbunet=SRLFBUNet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=1, act_mode='R', 
                #  downsample_mode="strideconv", upsample_mode="convtranspose")
        # SIMDB+UNet
        # self.drunet = DResUNet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=2, act_mode='R', 
                #  downsample_mode="strideconv", upsample_mode="convtranspose")
        self.drunet = SIMDUNet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=2, act_mode=self.act_mode, 
                 downsample_mode="strideconv", upsample_mode="convtranspose")
        #Res+Unet
        # self.drunet =UNetRes(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=2, act_mode='R', 
                #  downsample_mode="strideconv", upsample_mode="convtranspose")
        
        # self.rdunet = RDUNet(inchannels=4,outchannels=3,base_filters=64)
        # self.rgb_final = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)
        # self.w_final = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1)
        # self.final = nn.Conv2d(in_channels=80,out_channels=4,kernel_size=3,padding=1)
        # self.w_up = common.ConvBlock(in_channelss= 1,out_channels= 3,kernel_size= 1 , bias=False,act_type='relu')
        # self.pac = PacConv2d(in_channels=3, out_channels=3, kernel_size=5,stride=1,padding=2)
        
    def forward(self, x):
        rgb_x=x[:,0:3,:,:]
        w_x=x[:,3:4,:,:]
        rgb_x1=self.rgb_head(rgb_x)
        w_x1=self.w_head(w_x)
        # print(rgb_x1.shape)
        
        # '''
        #堆叠res block
        rgb_x1=self.rgb_res_block(rgb_x1)
        w_x1=self.w_res_block(w_x1)

        rgb_x1=self.rgb_res_block(rgb_x1)
        w_x1=self.w_res_block(w_x1)

        rgb_x1=self.rgb_res_block(rgb_x1)
        w_x1=self.w_res_block(w_x1)

        rgb_x1=self.rgb_res_block(rgb_x1)
        w_x1=self.w_res_block(w_x1)
        
        # rgb_x1=self.rgb_res_block(rgb_x1)
        # rgb_x1=self.rgb_res_block(rgb_x1)
        # '''
        
        '''
        #堆叠AdaFeatBlock
        rgb_x1=self.rgb_AFeB(rgb_x1)
        w_x1=self.w_AFeB(w_x1)
        
        rgb_x1=self.rgb_AFeB(rgb_x1)
        w_x1=self.w_AFeB(w_x1)
        '''
        # fusion = self.pac(rgb_x1,w_x1)
        # nums_channels=torch.cat((rgb_x1,w_x1),dim=1)
        # rgbw=self.final(nums_channels)
        # rgb_output=self.rgb_final(rgb_x1)
        # w_output = self.w_final(w_x1)
        # rgbw=torch.cat((rgb_output,w_output),dim=1)
        # fusion = self.pac(rgb_output,w_output)
        # output=self.unetmyself(fusion,1)
        
        rgb_x2=rgb_x1+rgb_x
        w_x2=w_x1+w_x
        # w_up=self.w_up(w_x2)
        fusion=torch.cat((rgb_x2,w_x2),dim=1)
        # fusion=self.pac(rgb_x2,w_x2)
        # output=self.rlfbunet(fusion)
        # output=self.srlfbunet(fusion)
        output=self.drunet(fusion)
        # output = self.rdunet(fusion)
        return output
    
    
    
