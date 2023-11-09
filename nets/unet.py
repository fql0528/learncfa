'''
U-Net implementation from Julien Martel
Edited by Cindy Nguyen
'''
import torch.nn.functional as F
import nets.modules as blocks
import torch.nn as nn
import nets.utils


class UNet(nn.Module):
    def __init__(self,
                 in_ch=1,
                 out_ch=1,
                 depth=3,
                 wf=3,
                 padding=True,
                 batch_norm=True,
                 up_mode='upsample'):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')

        prev_channels = in_ch

        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(blocks.MiniConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)) # 原版 used MINICONV BLOCK
            # self.encoders.append(blocks.MyMiniConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)) # 非原版
            prev_channels = 2 ** (wf + i)
        # print('self.encoders',self.encoders)

        self.decoders = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoders.append(blocks.UpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)
        # print('self.decoders',self.decoders)
        self.final_conv = nn.Conv2d(prev_channels, out_ch, kernel_size=1)
        self.clamp = nets.utils.clamp_class.apply


    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.encoders):
            # print('i',i)
            # print('down',down)
            x = down(x)
            if i != len(self.encoders) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
            
        for i, up in enumerate(self.decoders):
            # print('i',i)
            # print('up',up)
            x = up(x, blocks[-i - 1])

        x = self.final_conv(x)
        x = self.clamp(x)

        return x

