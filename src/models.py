import torch.nn as nn
import shutters.shutters as shutters
import torch
from nets.unet import UNet
from nets.mprnet import MPRNet_s2
from nets.transform_modules import TileInterp, TreeMultiRandom
from nets.dncnn import DnCNN, init_weights
from PConv.PConv import PartialConv2d
from nets.unet_myself import UNetMyself

# device = 'cuda:0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def define_shutter(shutter_type, args, test=False, model_dir=''):
    # print('args.block_size',args.block_size)
    return shutters.Shutter(shutter_type=shutter_type, block_size=args.block_size,
                            test=test, resume=args.resume, model_dir=model_dir, init=args.init)

def define_myshutter(shutter_type, args,block_size, test=False, model_dir=''):
    return shutters.Shutter(shutter_type=shutter_type, block_size=block_size,
                            test=test, resume=args.resume, model_dir=model_dir, init=args.init)


def define_model(shutter, decoder, args, get_coded=False):
    ''' Define any special interpolation modules in between encoder and decoder
    在编码器和解码器之间定义任何特殊的插值模块'''
    # print('args.shutter',args.shutter) #lsvpe
    if args.shutter in ['short', 'med', 'long', 'full'] or decoder is None or args.interp is None:
        print('***** No interpolation module added 未添加插补模块! *****')
        return Model(shutter, decoder, dec_name=args.decoder, get_coded=get_coded)
    elif args.shutter == 'quad' in args.shutter:
        if args.interp == 'bilinear':
            print('***** Bilinear 2x2 interpolation module added增加双线性2x2插补模块! *****')
            return TileInterpModel(shutter, decoder,
                                   get_coded=get_coded,
                                   shutter_name=args.shutter,
                                   sz=args.block_size[-1])
        elif args.interp == 'scatter':
            print('***** Scatter 2x2 interpolation module added!分散2x2插值模块添加! *****')
            return TreeModel(shutter,
                             decoder,
                             shutter_name=args.shutter,
                             get_coded=get_coded,
                             sz=[args.block_size[-2],args.block_size[-1]], #原版sz=args.block_size[-1],
                             k=args.k,
                             p=args.p,
                             num_channels=4)

    elif args.shutter == 'nonad' in args.shutter:
        if args.interp == 'bilinear':
            print('***** Bilinear 3x3 interpolation module added!双线性3x3插补模块添加! *****')
            return TileInterpModel(shutter, decoder,
                                   get_coded=get_coded,
                                   shutter_name=args.shutter,
                                   sz=args.block_size[-1])
        elif args.interp == 'scatter':
            print('***** Scatter 3x3 interpolation module added散点3x3插值模块增加! *****')
            return TreeModel(shutter,
                             decoder,
                             shutter_name=args.shutter,
                             get_coded=get_coded,
                             sz=[args.block_size[-2],args.block_size[-1]], #原版sz=args.block_size[-1],
                             k=args.k,
                             p=args.p,
                             num_channels=9)
    elif args.interp == 'scatter' and args.shutter in ['lsvpe', 'uniform', 'poisson']:
        return TreeModel(shutter,
                         decoder,
                         shutter_name=args.shutter,
                         get_coded=get_coded,
                         k=args.k,
                         p=args.p,
                         num_channels=9,
                         sz=[args.block_size[-2],args.block_size[-1]])#原版sz=args.block_size[-1],
    elif args.interp=='pconv' and args.shutter=='rgbw':
        print('*******添加PConvModel*********')
        return PConvModel(shutter,decoder,shutter_name=args.shutter)

    elif args.interp=='scatter'and args.shutter=='rgbw':
        print("******添加TreeRGBWModel******")
        return TreeRGBWModel(shutter,
                         decoder,
                         shutter_name=args.shutter,
                         get_coded=get_coded,
                         k=args.k,
                         p=args.p,
                         num_channels=4,
                         sz=[args.block_size[-2],args.block_size[-1]])#原版sz=args.block_size[-1],
    raise NotImplementedError('Interp + Shutter combo has not been implemented')


def define_decoder(model_name, args):
    if args.decoder == 'none':
        return None
    out_ch = 1
    if args.shutter == 'full':
        in_ch = 3
    elif args.shutter in ['short', 'med', 'long'] or args.interp is None:
        in_ch = 1
    elif args.shutter=='rgbw' and args.interp=='scatter':
        in_ch=4
        out_ch=4
    # elif args.shutter=='lsvpe' and args.interp=='scatter':
    #     in_ch=9
    elif 'quad' in args.shutter:
        in_ch = 4
    elif 'nonad' in args.shutter:
        in_ch = 9
    elif args.interp == 'scatter':
        in_ch = 9
    elif args.interp=='pconv':
        in_ch=4
        out_ch=4
    else:
        raise NotImplementedError
    # print('in_ch',in_ch) #4
    # print('out_ch',out_ch) #4
    if model_name == 'unet':
        #对于RGBW必须要加BatchNormal层，否则没有效果
        return UNet(in_ch=in_ch, out_ch=out_ch, depth=6, wf=5, padding=True, batch_norm=False, up_mode='upconv') #upsample 原版up_mode='upconv' batch_norm=False
    if model_name == 'mpr':
        return MPRNet_s2(in_c=in_ch)
    if model_name == 'dncnn':
        model = DnCNN(in_nc=in_ch, out_nc=1, nc=64, nb=17, act_mode='BR', shutter_name=args.shutter)
        init_weights(model, init_type='orthogonal', init_bn_type='uniform', gain=0.2)
        return model
    if model_name=='myunet':
        return UNetMyself(n_channels=in_ch,n_classes=out_ch)
    raise NotImplementedError('Model not specified correctly')


class TreeModel(nn.Module):
    def __init__(self, shutter, decoder, get_coded=False,
                 shutter_name=None, sz=512, k=3, p=1, num_channels=8):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        # print('self.shutter',self.shutter) # LSVPE() Quad()
        self.decoder = decoder
        self.shutter_name = shutter_name
        self.tree = TreeMultiRandom(sz=sz, k=k, p=p, num_channels=num_channels)

    def forward(self, input, train=True):
        # print('input',input)
        # print('input.shape',input.shape) #torch.Size([2, 8, 512, 512])   Quad torch.Size([2, 8, 512, 512])
        # coded,end_params = self.shutter(input, train=train) #测试梯度
        coded = self.shutter(input, train=train)
        # print(coded==0)
        # print('self.shutter',self.shutter) #Quad()
        # print('coded',coded)1
        # print('coded.shape',coded.shape) #torch.Size([2, 1, 512, 512])   Quad torch.Size([2, 1, 512, 512])
        # print('self.shutter.getLength().shape',self.shutter.getLength().shape)
        multi = self.tree(coded, self.shutter.getLength())
        # print('multi.shape',multi.shape) #torch.Size([1, 4, 512, 512])
        # print('multi',multi)
        # print('multi.shape',multi.shape) #torch.Size([2, 9, 512, 512])   Quad torch.Size([2, 4, 512, 512])
        x = self.decoder(multi) 
        # x = self.decoder(coded) 
        # print('x.shape',x.shape)#torch.Size([2, 1, 512, 512])            Quad torch.Size([2, 1, 512, 512])
        if self.get_coded:
            return x, coded
        # return x,end_params #测试梯度
        return x
        # return multi

    def forward_using_capture(self, coded):
        multi = self.tree(coded, self.shutter.getLength())
        x = self.decoder(multi)
        if self.get_coded:
            return x, coded
        return x

class TreeRGBWModel(nn.Module):
    def __init__(self, shutter, decoder, get_coded=False,
                 shutter_name=None, sz=512, k=3, p=1, num_channels=8):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        # print('self.shutter',self.shutter) # LSVPE() Quad()
        self.decoder = decoder
        self.shutter_name = shutter_name
        self.tree = TreeMultiRandom(sz=sz, k=k, p=p, num_channels=num_channels)

    def forward(self, input, train=True):
        # print('input.shape',input.shape) #RGBW:torch.Size([1, 4, 512, 512])  torch.Size([2, 8, 512, 512])   Quad torch.Size([2, 8, 512, 512])
        coded,mask = self.shutter(input, train=train)
        # print('coded.shape',coded.shape) #RGBW torch.Size([1, 4, 512, 512])  torch.Size([2, 1, 512, 512])   Quad torch.Size([2, 1, 512, 512])   RGBWtorch.Size([1, 4, 512, 512])
        multi = self.tree(coded, self.shutter.getLength())
        # print('multi.shape',multi.shape) #RGBW torch.Size([1, 4, 512, 512]) torch.Size([2, 9, 512, 512])   Quad torch.Size([2, 4, 512, 512])
        x = self.decoder(multi)
        # print('x.shape',x.shape)#RGBW torch.Size([1, 4, 512, 512])  torch.Size([2, 1, 512, 512])            Quad torch.Size([2, 1, 512, 512])
        if self.get_coded:
            return x, coded
        return x

    def forward_using_capture(self, coded):
        multi = self.tree(coded, self.shutter.getLength())
        x = self.decoder(multi)
        if self.get_coded:
            return x, coded
        return x

class TileInterpModel(nn.Module):
    def __init__(self, shutter, decoder, get_coded=False,
                 shutter_name='nonad', sz=512, interp='bilinear'):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder

        if 'nonad' in shutter_name:
            self.tile_size = 3
        elif 'quad' in shutter_name:
            self.tile_size = 2
        else:
            raise NotImplementedError
        self.interpolator = TileInterp(shutter_name=shutter_name, tile_size=self.tile_size, sz=sz, interp=interp)

    def forward(self, input, train=True):
        # print('input.shape',input.shape) #torch.Size([2, 8, 512, 512])
        coded = self.shutter(input, train=train)
        # print('coded',coded)
        # print(torch.min(coded)) #0
        # print('coded.shape',coded.shape) #torch.Size([2, 1, 512, 512])
        multi = self.interpolator(coded)
        # print('multi.shape',multi.shape) #torch.Size([2, 4, 512, 512])
        # print(torch.min(multi)) #0
        x = self.decoder(multi)
        # print('x.shape',x.shape) #torch.Size([2, 1, 512, 512])
        if self.get_coded:
            return x, multi
        return x

    def forward_using_capture(self, coded):
        # print("#################") #没输出
        multi = self.interpolator(coded)
        x = self.decoder(multi)
        if self.get_coded:
            return x, multi
        return x

class PConvModel(nn.Module):
    def __init__(self, shutter, decoder,  shutter_name='rgbw'):
        super(PConvModel, self).__init__()
        self.shutter = shutter
        self.decoder = decoder
        # print('self.shutter',self.shutter) # RGBW()
        # print('self.decoder',self.decoder) #UNet(.....


        #定义卷积层
        self.conv=PartialConv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False, multi_channel = True,return_mask=True).to(device) #官方
        # 定义unet层
        # self.unet=UNet(n_channels=3,n_classes=3).to(device)
    def forward(self,image,train=True):
        rgbw_raw,mask=self.shutter(image)
        # print('rgbw_raw.shape',rgbw_raw.shape) #torch.Size([1, 4, 512, 512])
        # print('mask.shape',mask.shape) #torch.Size([1, 4, 512, 512])
        # print('mask',mask)
        rgbw,update_mask=self.conv(rgbw_raw,mask)
        # print('rgbw.shape',rgbw.shape) #torch.Size([1, 4, 512, 512])
        output=self.decoder(rgbw)
        # print('output.shape',output.shape) #torch.Size([1, 1, 512, 512])
        # output=self.unet(image_output)
        return output

class Model(nn.Module):
    def __init__(self, shutter, decoder, dec_name=None, get_coded=False):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder
        self.dec_name = dec_name

    def forward(self, input, train=True):
        coded = self.shutter(input, train=train)
        if not coded.requires_grad:
            ## needed for computing gradients wrt input for fixed shutters
            coded.requires_grad = True
        if self.decoder is None:
            if self.get_coded:
                return coded, coded
            return coded

        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x

    def forward_using_capture(self, coded):
        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x
