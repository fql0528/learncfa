import torch.nn as nn
import nets.common as common
# from nets.sgnet import m_res

class model_res(nn.Module):  #去马赛克相关模型
    # def __init__(self, opt):
    def __init__(self,channels,act_type,bias,norm_type):
        super(model_res, self).__init__()
        dm_n_feats = channels
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


class RRDBNet(nn.Module):
    def __init__(self,channels,act_type,bias,norm_type):
        super(RRDBNet, self).__init__()
        dm_n_feats=channels
        
        self.rrdbnet=model_res(channels, act_type, bias, norm_type)
        self.conv_head=common.ConvBlock(in_channelss=4,out_channels=dm_n_feats,kernel_size=5,
                                      act_type=act_type, bias=bias)
        
        self.conv_final=common.ConvBlock(in_channelss=dm_n_feats,
                                         out_channels=3,kernel_size=3, bias=bias)

    def forward(self,x):
        x1=self.conv_head(x)
        x2=self.rrdbnet(x1)
        x3=self.conv_final(x1+x2)
        
        return x3
        
        
        
        
        