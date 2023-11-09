from torch import nn
from CFA.TreeMultiRandom import TreeMultiRandom
from nets.unet_myself import UNetMyself

class TreeMulti_Unet_connect(nn.Module):
    def __init__(self,device,**kwargs):
        super(TreeMulti_Unet_connect, self).__init__(**kwargs)
        #定义插值
        self.tree = TreeMultiRandom(k=3, p=1.0, num_channels=3).to(device)
        #定义卷积层
        # self.conv=PartialConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False,groups=3, multi_channel = True,return_mask=True).to(device) #官方
        #定义unet层
        self.unet=UNetMyself(n_channels=3,n_classes=3).to(device)
    def forward(self,image):
        image_output=self.tree(image)
        # image_output,_=self.conv(image,mask)
        output=self.unet(image_output)
        return output