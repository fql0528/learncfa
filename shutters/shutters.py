import torch
import torch.nn as nn
import os
import shutters.shutter_utils as shutils
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

# repo_dir = '/home/cindy/PycharmProjects/coded-deblur-publish/shutters/shutter_templates'
repo_dir='../shutters/shutter_templates'

def add_noise(img, exp_time=1, test=False):
    sig_shot_min = 0.0001
    sig_shot_max = 0.01
    sig_read_min = 0.001
    sig_read_max = 0.03
    if test: # keep noise levels fixed when testing 测试时保持固定的噪音水平
        sig_shot = (sig_shot_max - sig_shot_min) / 2
        sig_read = (sig_read_max - sig_read_min) / 2
    else:
        sig_shot = (sig_shot_min - sig_shot_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_shot_max
        sig_read = (sig_read_min - sig_read_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_read_max

    ratio = exp_time / 8

    # Scale image corresponding to exposure time
    img = img * ratio

    # Add shot noise, must detach or it'll mess with the computational graph
    shot = (img.detach() ** (1/2)) * sig_shot * torch.randn_like(img)

    # Add read noise. Short and long exposures should have the same read noise.
    read = sig_read * torch.randn_like(img)
    return img + shot + read


class Shutter:
    def __new__(cls, shutter_type, block_size, test=False, resume=False, model_dir='', init='even'):
        cls_out = {
            'short': Short,
            'long': Long,
            'quad': Quad,
            'med': Medium,
            'full': Full,
            'uniform': Uniform,
            'poisson': Poisson,
            'nonad': Nonad,
            'lsvpe': LSVPE,
            'rgbw':RGBW,
        }[shutter_type]

        return cls_out(block_size, test, resume, model_dir, init)


class ShutterBase(nn.Module):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__()
        self.block_size = block_size
        self.test = test
        self.resume = resume
        self.model_dir = os.path.dirname(model_dir) # '21-11-14/21-11-14-net/short'

    def getLength(self):
        raise NotImplementedError('Must implement in derived class')

    def getMeasurementMatrix(self):
        raise NotImplementedError('Must implement in derived class')

    def forward(self, video_block, train=True):
        raise NotImplementedError('Must implement in derived class')

    def post_process(self, measurement, exp_time, test):  #exp_time：表示曝光时间
        # measurement.shape torch.Size([2, 1, 512, 512])
        #  exp_time.shape  torch.Size([1,512,512])
        measurement = torch.div(measurement, exp_time)       # 1 1 H W
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        measurement = add_noise(measurement, exp_time=exp_time, test=test)
        measurement = torch.clamp(measurement, 0, 1)  # 把measurement归一化到0--1
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        return measurement

    def count_instances(self, lengths, counts):
        flattened_lengths = lengths.reshape(-1, ).type(torch.int8)
        total_counts = torch.bincount(flattened_lengths).cpu()
        for k in range(1, len(total_counts)):
            counts[k] = total_counts[k]
        return counts


class Short(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

    def getLength(self):
        return torch.ones((1, 1, self.block_size[-2], self.block_size[-1]), dtype=torch.float32)

    def forward(self, video_block, train=True):
        # print('video_block.shape',video_block.shape)  #torch.Size([2, 8, 512, 512])
        measurement = video_block[:, :1, ...]
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        measurement = self.post_process(measurement, exp_time=1, test=self.test)
        return measurement


class Medium(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

    def getLength(self):
        return torch.ones((1, 1, self.block_size[-2], self.block_size[-1]), dtype=torch.float32) * 4

    def forward(self, video_block, train=True):
        # print('video_block.shape',video_block.shape)  #torch.Size([2, 8, 512, 512])
        measurement = torch.sum(video_block[:, :4, ...], dim=1, keepdim=True)
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        measurement = self.post_process(measurement, exp_time=4, test=self.test) #exp_time：曝光时间
        return measurement


class Long(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size

    def getLength(self):
        return torch.ones((1, 1, self.block_size[-2], self.block_size[-1]), dtype=torch.float32) * 8

    def forward(self, video_block, train=True):
        measurement = torch.sum(video_block[:, :, ...], dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=8, test=self.test)
        return measurement


class Full(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)

    def forward(self, video_block, train=True):
        measurement = video_block[:, :1, ...]
        short = self.post_process(measurement, exp_time=1, test=self.test)
        # print('short.shape',short.shape)#torch.Size([2, 1, 512, 512])   

        measurement = torch.sum(video_block[:, :4, ...], dim=1, keepdim=True)
        med = self.post_process(measurement, exp_time=4, test=self.test)
        # print('med.shape',med.shape) #torch.Size([2, 1, 512, 512])   

        measurement = torch.sum(video_block[:, :, ...], dim=1, keepdim=True)
        long = self.post_process(measurement, exp_time=8, test=self.test)
        # print('long.shape',long.shape) #torch.Size([2, 1, 512, 512])   
        # print('torch.cat((long, med, short), dim=1).shape',torch.cat((long, med, short), dim=1).shape) #torch.Size([2, 3, 512, 512])   
        return torch.cat((long, med, short), dim=1)


class Quad(ShutterBase):
    ''' Design consistent with Jiang et al. HDR reconstruction, exposure ratios are 1:4:8
    设计与Jiang等一致。HDR重建，曝光比为1:4:8
    '''
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)

        self.shutter = torch.zeros(self.block_size, dtype=torch.float32, device=device)
        # print('##################################')
        # print('self.shutter.shape',self.shutter.shape) #orch.Size([8, 512, 512])
        half_h = int(self.block_size[1] / 2)
        half_w = int(self.block_size[2] / 2)

        self.shutter[:, 0::2, 0::2] = torch.ones((8, half_h, half_w), device=device)
        # print('self.shutter',self.shutter)
        self.shutter[:4, 1::2, 0::2] = torch.ones((4, half_h, half_w), device=device)
        # print('self.shutter',self.shutter)
        self.shutter[:4, 0::2, 1::2] = torch.ones((4, half_h, half_w), device=device)
        # print('self.shutter',self.shutter)
        self.shutter[:1, 1::2, 1::2] = torch.ones((1, half_h, half_w), device=device)
        self.shutter = self.shutter.unsqueeze(0)
        # print('self.shutter',self.shutter)
        # print('self.shutter.shape',self.shutter.shape) # torch.Size([1, 8, 512, 512])

        self.num_frames = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32, device=device)
        self.num_frames[0::2, 0::2] *= 8
        self.num_frames[1::2, 0::2] *= 4
        self.num_frames[0::2, 1::2] *= 4
        # print('self.num_frames',self.num_frames)
        self.num_frames = self.num_frames.unsqueeze(0) #torch.Size([1,512,512])

    def getLength(self):
        # print('self.num_frames.shape',self.num_frames.shape) #torch.Size([1, 512, 512])
        # print('self.num_frames',self.num_frames) #
        """
        tensor([[[8., 4., 8.,  ..., 4., 8., 4.],
         [4., 1., 4.,  ..., 1., 4., 1.],
         [8., 4., 8.,  ..., 4., 8., 4.],
         ...,
         [4., 1., 4.,  ..., 1., 4., 1.],
         [8., 4., 8.,  ..., 4., 8., 4.],
         [4., 1., 4.,  ..., 1., 4., 1.]]], device='cuda:0')
        """
        
        return self.num_frames

    def forward(self, video_block, train=True):
        print("###################")
        # print('video_block',video_block)
        # print('video_block.shape',video_block.shape) #torch.Size([2, 8, 512, 512])
        print('self.shutter.shape',self.shutter.shape) #torch.Size([1, 8, 512, 512])
        print('self.shutter',self.shutter)
        measurement = torch.mul(self.shutter, video_block)          # 1 8 H W
        print('measurement',measurement)
        print('measurement.shape',measurement.shape) #torch.Size([2, 8, 512, 512])
        measurement = torch.sum(measurement, dim=1, keepdim=True)   # 1 1 H W
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        measurement = self.post_process(measurement, exp_time=self.num_frames, test=self.test)
        # print('measurement.shape',measurement.shape) #torch.Size([2, 1, 512, 512])
        return measurement


class Uniform(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.lengths = torch.randint(1, block_size[0] + 1, size=(block_size[1], block_size[2]), device=device)
        # print('self.lengths.shape',self.lengths.shape) #torch.Size([512, 512])
        # print('self.lengths',self.lengths)
        """
        tensor([[5, 6, 1,  ..., 5, 4, 2],
        [4, 7, 8,  ..., 8, 1, 8],
        [5, 1, 6,  ..., 7, 2, 8],
        ...,
        [8, 1, 8,  ..., 5, 1, 4],
        [5, 1, 8,  ..., 4, 8, 8],
        [8, 6, 1,  ..., 8, 6, 2]], device='cuda:0')
        """
        self.shutter = torch.zeros(block_size, device=device)

        for i in range(block_size[0] + 1):
            (y, x) = (self.lengths == i).nonzero(as_tuple=True)
            self.shutter[:i, y, x] = torch.ones((i, len(y)), device=device)

    def getLength(self):
        return self.lengths

    def forward(self, video_block, train=True):
        measurement = torch.mul(video_block, self.shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement


class Poisson(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.lengths = torch.load(f'{repo_dir}/poisson.pt') + 1.0
        if block_size[-1] != 512:
            self.lengths = self.lengths[..., :block_size[-2], :block_size[-1]]
        self.lengths = self.lengths.to(device)

        self.shutter = torch.zeros(block_size, device=device)
        for i in range(block_size[0] + 1):
            (y, x) = (self.lengths == i).nonzero(as_tuple=True)
            self.shutter[:i, y, x] = torch.ones((i, len(y)), device=device)

    def getLength(self):
        return self.lengths

    def forward(self, video_block, train=True):
        measurement = torch.mul(video_block, self.shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement


class Nonad(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        # load preinitialized shutter, must add 1 to get correct lengths between [1-8]
        self.lengths = torch.load(f'{repo_dir}/nonad.pt') + 1.0
        if block_size[-1] != 512:
            self.lengths = self.lengths[..., :block_size[-2], :block_size[-1]]
        self.lengths = self.lengths.to(device)

        self.shutter = torch.zeros(block_size, device=device)
        for i in range(1, block_size[0] + 1):
            (y, x) = (self.lengths == i).nonzero(as_tuple=True)
            self.shutter[:i, y, x] = torch.ones((i, len(y)), device=device)

    def getLength(self):
        return self.lengths

    def forward(self, video_block, train=True):
        measurement = torch.mul(video_block, self.shutter)
        # print('self.shutter',self.shutter)
        # print('self.shutter.shape',self.shutter.shape)
        # print('measurement',measurement)
        # print('measurement.shape',measurement.shape)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        # print('measurement',measurement)
        # print('measurement.shape',measurement.shape)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        return measurement


class LSVPE(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size
        # print('init',init) #quad
        if init == 'even':
            rand_end = self.block_size[0] * torch.rand((self.block_size[1], self.block_size[2]), dtype=torch.float32)
        elif init == 'ones':
            rand_end = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32)
        elif init == 'quad':
            rand_end = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32)
            rand_end[::2, ::2] *= 8.0
            # print('rand_end1',rand_end)
            rand_end[1::2, ::2] *= 4.0
            # print('rand_end2',rand_end)
            rand_end[::2, 1::2] *= 4.0
            # print('rand_end3',rand_end)
            """
            tensor([[8., 4., 8.,  ..., 4., 8., 4.],
        [4., 1., 4.,  ..., 1., 4., 1.],
        [8., 4., 8.,  ..., 4., 8., 4.],
        ...,
        [4., 1., 4.,  ..., 1., 4., 1.],
        [8., 4., 8.,  ..., 4., 8., 4.],
        [4., 1., 4.,  ..., 1., 4., 1.]])
            """
        else:
            raise NotImplementedError

        self.end_params = nn.Parameter(rand_end, requires_grad=True)
        # print('self.end_params.shape',self.end_params.shape) #torch.Size([512, 512])
        # print('self.end_params',self.end_params)
        """Parameter containing:
        tensor([[8., 4., 8.,  ..., 4., 8., 4.],
        [4., 1., 4.,  ..., 1., 4., 1.],
        [8., 4., 8.,  ..., 4., 8., 4.],
        ...,
        [4., 1., 4.,  ..., 1., 4., 1.],
        [8., 4., 8.,  ..., 4., 8., 4.],
        [4., 1., 4.,  ..., 1., 4., 1.]], requires_grad=True)
        """
        

        self.time_range = torch.arange(0, self.block_size[0], dtype=torch.float32, device=device)[:, None, None]
        # print('self.time_range',self.time_range) #torch.Size([8, 1, 1])
        # print('self.time_range.shape',self.time_range.shape)
        self.time_range = self.time_range.repeat(1, self.block_size[1], self.block_size[2])
        # print('self.time_range1',self.time_range)
        # print('self.time_range1.shape',self.time_range.shape) # torch.Size([8, 512, 512])
        self.total_steps = 0
        self.lengths = torch.zeros((self.block_size[1], self.block_size[2]))
        self.total_steps = 0
        self.counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    def getLength(self):
        end_params_int = torch.clamp(self.end_params, 1.0, self.block_size[0])
        shutter = shutils.less_than(self.time_range, end_params_int)
        self.lengths = torch.sum(shutter, dim=0)
        # print("111111",self.lengths.shape) #torch.Size([512, 512])

        self.lengths=self.lengths.unsqueeze(0) #非原版
        # print('2222',self.lengths.shape) #torch.Size([1, 512, 512])
        return self.lengths

    def forward(self, video_block, train=False):
        if train:
            self.total_steps += 1

#clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        end_params_int = torch.clamp(self.end_params, 1.0, self.block_size[0])
        # print('self.end_params',self.end_params)
        # print('self.end_params.grad',self.end_params.grad)
        # print('end_params_int',end_params_int)
        """
        tensor([[8., 4., 8.,  ..., 4., 8., 4.],
        [4., 1., 4.,  ..., 1., 4., 1.],
        [8., 4., 8.,  ..., 4., 8., 4.],
        ...,
        [4., 1., 4.,  ..., 1., 4., 1.],
        [8., 4., 8.,  ..., 4., 8., 4.],
        [4., 1., 4.,  ..., 1., 4., 1.]], device='cuda:0',
       grad_fn=<ClampBackward>)   会随着训练进行更新
        """
        # print('end_params_int.shape',end_params_int.shape) # torch.Size([512, 512])
        # print('self.time_range.shape',self.time_range.shape)#torch.Size([8, 512, 512])
        # print('self.time_range',self.time_range)
        shutter = shutils.less_than(self.time_range, end_params_int)
        # print('shutter',shutter)
        self.lengths = torch.sum(shutter, dim=0)
        # print('self.lengths',self.lengths)

        if train and self.total_steps % 200 == 0:
            self.counts = self.count_instances(self.lengths, self.counts)
        # print('video_block.shape',video_block.shape) #torch.Size([2, 8, 512, 512])     
        # print('shutter.shape',shutter.shape) #torch.Size([8, 512, 512])
        measurement = torch.mul(video_block, shutter)
        measurement = torch.sum(measurement, dim=1, keepdim=True)
        measurement = self.post_process(measurement, exp_time=self.lengths, test=self.test)
        # print('measurement.shape',measurement.shape)#torch.Size([1, 1, 512, 512])
        # measurement.retain_grad()
        return measurement

class RGBW(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.length = torch.ones((self.block_size[1], self.block_size[2]), dtype=torch.float32, device=device)
        self.length[::4,2::4]=1
        self.length[1::4,3::4]=1
        self.length[::4,::4]=2
        self.length[1::4,1::4]=2
        self.length[2::4,2::4]=2
        self.length[3::4,3::4]=2
        self.length[2::4,::4]=3
        self.length[3::4,1::4]=3
        self.length[::2,1::2]=4
        self.length[1::2,::2]=4
        # print('self.length',self.length)
        # print('self.length.shape',self.length.shape) #torch.Size([512, 512])
        self.length = self.length.unsqueeze(0)  #torch.Size([1,512, 512])
    def getLength(self):
        return self.length
    def forward(self,img_rgbw,train=True):
        #模拟RGBW格式图像
        #针对RGB格式 是[通道,高，宽]
        mask = torch.zeros(img_rgbw.shape)
        # print('mask',mask)
        # print('mask.shape',mask.shape) #mask.shape (1,4, 512, 512)
        # red channels
        mask[:,0,::4,2::4]=1
        mask[:,0,1::4,3::4]=1
        #green channels
        mask[:,1,::4,::4]=1
        mask[:,1,1::4,1::4]=1
        mask[:,1,2::4,2::4]=1
        mask[:,1,3::4,3::4]=1
        #blue channels
        mask[:,2,2::4,::4]=1
        mask[:,2,3::4,1::4]=1
        #W channel
        mask[:,3,::2,1::2]=1
        mask[:,3,1::2,::2]=1
        mask=mask.to(device)
        bayer_rgbw = mask * img_rgbw
        # print('bayer_rgbw.shape',bayer_rgbw.shape) #torch.Size([1, 4, 512, 512])
        # print('mask.shape',mask.shape) #torch.Size([1, 4, 512, 512])
        return bayer_rgbw,mask
