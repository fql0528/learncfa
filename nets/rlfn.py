# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import torch.nn as nn
# from model import block
from nets import rlfn_block


class RLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 upscale=4):
        super(RLFN, self).__init__()

        self.conv_1 = rlfn_block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = rlfn_block.RLFB(feature_channels)
        self.block_2 = rlfn_block.RLFB(feature_channels)
        self.block_3 = rlfn_block.RLFB(feature_channels)
        self.block_4 = rlfn_block.RLFB(feature_channels)
        self.block_5 = rlfn_block.RLFB(feature_channels)
        self.block_6 = rlfn_block.RLFB(feature_channels)

        self.conv_2 = rlfn_block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)
        self.conv_1x1=rlfn_block.conv_layer(feature_channels,out_channels,kernel_size=1)

        # self.upsampler = rlfn_block.pixelshuffle_block(feature_channels,
                                                #   out_channels,
                                                #   upscale_factor=upscale)  #原版上采样，本项目不需要

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        # output = self.upsampler(out_low_resolution) #原版
        output = self.conv_1x1(out_low_resolution)

        return output

class SRLFB(nn.Module):
    """
    Sample Residual Local Feature Block (RLFB).
    去掉ESA模块
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.c5(out)

        return out