import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Residual-Dense U-net for image denoising

@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        # print('upsample11.shape',upsample.shape) #torch.Size([1, 1024, 255, 169])
        upsample = self.actv_t(self.conv_t(upsample))
        # print('concat.shape',concat.shape) #torch.Size([1, 512, 510, 339]) 
        # print('upsample.shape',upsample.shape) #torch.Size([1, 1024, 510, 338])
        # """
        if((upsample.shape[2]*2!=concat.shape[2]) or (upsample.shape[3]*2!=concat.shape[3])):
            # input is CHW
            diffY = concat.size()[2] - upsample.size()[2]
            diffX = concat.size()[3] - upsample.size()[3]
            upsample = F.pad(upsample, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # print('upsample22.shape',upsample.shape) #torch.Size([1, 512, 510, 339])
        # """
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x


class RDUNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, inchannels,outchannels,base_filters):
        super().__init__()

        # channels = kwargs['channels']
        # filters_0 = kwargs['base filters']
        in_channels=inchannels
        out_channels=outchannels
        filters_0=base_filters
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(in_channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, out_channels)

    def forward(self, inputs):
        # print('inputs.shape',inputs.shape) #torch.Size([1, 4, 2041, 1359])
        
        """
        #这里借鉴ResUnet的处理，记得上采样中相同功能关掉，只用其一即可
        #记得和下面的return同步
        #这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
        # print('inputs.shape',inputs.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
        h, w = inputs.size()[-2:]
        # print('h',h) #128  2041
        # print('w',w) #128  1359
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        # print('paddingBottom',paddingBottom) #0  7
        # print('paddingRight',paddingRight) #0 1
        inputs = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(inputs)
        # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
        # exit()
        """
        
        out_0 = self.input_block(inputs)    # Level 0
        # print('out_0.shape',out_0.shape) #torch.Size([1, 128, 2041, 1359])
        out_0 = self.block_0_0(out_0)
        # out_0 = self.block_0_1(out_0)
        # print('out_0.shape',out_0.shape) #torch.Size([1, 128, 2041, 1359])

        out_1 = self.down_0(out_0)          # Level 1
        # print('out_1.shape',out_1.shape) #torch.Size([1, 256, 1020, 679])
        out_1 = self.block_1_0(out_1)
        # out_1 = self.block_1_1(out_1)
        # print('out_1.shape',out_1.shape) #torch.Size([1, 256, 1020, 679])

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
        # out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
        # out_3 = self.block_3_1(out_3)

        # print('out_3.shape',out_3.shape) #torch.Size([1, 1024, 255, 169])
        # print('out_2.shape',out_2.shape) #torch.Size([1, 512, 510, 339]) 
        out_4 = self.up_2([out_3, out_2])   # Level 2
        out_4 = self.block_2_2(out_4)
        # out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])   # Level 1
        out_5 = self.block_1_2(out_5)
        # out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])   # Level 0
        out_6 = self.block_0_2(out_6)
        # out_6 = self.block_0_3(out_6)

        # return self.output_block(out_6) + inputs #原版
        # return self.output_block(out_6) + inputs[:,0:3,:,:]
        output=self.output_block(out_6) + inputs[:,0:3,:,:]
        # output=output[..., :h, :w]
        return output