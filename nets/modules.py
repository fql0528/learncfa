'''
Building blocks for U-Net written by Julien Martel U-Net的构建模块由Julien Martel编写
Edited by Cindy Nguyen Cindy Nguyen编辑
'''
import torch.nn as nn
import torch
import torch.nn.functional as F


class MiniConvBlock(nn.Module):
    '''
    Implements single conv + ReLU down block 实现单一的conv + ReLU下块
    '''
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        # print('int(padding)',int(padding)) #1
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class MyMiniConvBlock(nn.Module):
    '''
    Implements single conv + ReLU down block 实现单一的conv + ReLU下块
    '''
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        # blocks = []
        # print('int(padding)',int(padding)) #1
        # blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        # blocks.append(nn.ReLU())
        # if batch_norm:
            # blocks.append(nn.BatchNorm2d(out_ch))
        self.conv_block = MyConvBlock(in_ch, out_ch, padding, batch_norm) #非原版
        # self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # return self.blocks(x)
        return self.conv_block(x) #非原版

class MyConvBlock(nn.Module): #Conv2d--BatchNorm2d()--ReLU()
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        x=in_ch//2
        blocks.append(nn.Conv2d(in_ch, x, kernel_size=3, padding=int(padding))) #原版x=out_ch
        if batch_norm:
            blocks.append(nn.BatchNorm2d(x))
        blocks.append(nn.ReLU())

        blocks.append(nn.Conv2d(x, out_ch, kernel_size=3, padding=int(padding)))
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.ReLU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ConvBlock(nn.Module): #Conv2d--ReLU()--BatchNorm2d()
    def __init__(self, in_ch, out_ch, padding, batch_norm):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=int(padding)))
        blocks.append(nn.ReLU())
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) #原版
            )
        self.conv_block = ConvBlock(in_ch, out_ch, padding, batch_norm) #原版
        # self.conv_block = MyConvBlock(in_ch, out_ch, padding, batch_norm) #非原版

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_x + target_size[0]), diff_x:(diff_x + target_size[1])]
    def my_center_crop(self,x1,x2):
                # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1

    def forward(self, img, bridge):
        # print('img.shape',img.shape) #torch.Size([1, 1024, 16, 16])
        up = self.up(img)
        # print('up.shape',up.shape) #torch.Size([1, 512, 32, 32])
        crop = self.center_crop(bridge, up.shape[2:]) #原版
        out = torch.cat([up, crop], 1) #原版
        # print('out.shape',out.shape) #torch.Size([1, 1024, 32, 32])

        # crop=self.my_center_crop(up,bridge) #非原版
        # print('crop.shape',crop.shape) #crop.shape torch.Size([1, 512, 32, 32])
        # print('bridge.shape',bridge.shape) #torch.Size([1, 512, 32, 32])
        # out=torch.cat([bridge,crop],dim=1)#非原版
        # print('out.shape',out.shape) #torch.Size([1, 1024, 32, 32])

        return self.conv_block(out)# 原版


class ConvUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=9, padding=1, groups=in_ch)
            )
        self.conv_block = ConvBlock(in_ch, out_ch, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_x + target_size[0]), diff_x:(diff_x + target_size[1])]

    def upsample(self, layer):
        self.upsample_layer = nn.Upsample(mode='bilinear', scale_factor=2)



    def forward(self, img):
        up = self.up(img)
        return up
        # out = self.conv_block(up)

        # up = self.up(img)
        #
        # print(up.shape)
        # crop = self.center_crop(bridge, up.shape[2:])
        # print(crop.shape)
        # out = torch.cat([up, crop], 1)

        # return self.conv_block(out)