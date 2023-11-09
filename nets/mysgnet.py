import torch.nn as nn
import nets.common as common
from nets.network_unet import SIMDUNet
from nets.pac import PacConv2d

class MySGNet(nn.Module):#引导网络
    def __init__(self,in_nc=1, out_nc=1, 
                 nc=[64, 128, 256, 512], nb=4, act_mode='R', 
                 downsample_mode="strideconv", upsample_mode="convtranspose"):
        super(MySGNet, self).__init__()
        self.in_channels=in_nc
        self.out_channels=out_nc
        self.nc=nc
        self.nb=nb
        self.act_mode=act_mode
        self.downsample_mode=downsample_mode
        self.upsample_mode=upsample_mode
        self.w_dres_unet=SIMDUNet(in_nc=1,out_nc=1,nc=self.nc,nb=self.nb,act_mode=self.act_mode,
                              downsample_mode=self.downsample_mode,upsample_mode=self.upsample_mode)
        self.rgbw_dres_unet=SIMDUNet(in_nc=self.in_channels,out_nc=self.out_channels,nc=self.nc,nb=self.nb,act_mode=self.act_mode,
                              downsample_mode=self.downsample_mode,upsample_mode=self.upsample_mode)
        self.w_up = nn.Sequential(
                common.ConvBlock(in_channelss= 1,out_channels= 4,kernel_size= 1 , bias=True),
                # nn.LeakyReLU(0.2 , inplace = True), #原版
                nn.LeakyReLU(0.2 , inplace = False),
                # nn.ReLU(inplace=True),
                
                common.ConvBlock(in_channelss= 4,out_channels= 8,kernel_size= 1 , bias=True),
                # nn.LeakyReLU(0.2 , inplace = True) #原版
                nn.LeakyReLU(0.2 , inplace = False)
        )
        self.pac = PacConv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=5,stride=1,padding=2)
        self.final = common.ConvBlock(in_channelss= self.out_channels ,out_channels= 3,kernel_size= 3 , bias=True)
    def forward(self, x ):
        x1=x
        x=self.rgbw_dres_unet(x)
        w_output = self.w_dres_unet(x1[:,3:4,:,:].detach())
        w_combine = self.w_up(w_output)
        x = self.pac(x,w_combine)
        x = self.final(x)
        return x