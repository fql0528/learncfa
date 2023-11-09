
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from CFA.RGRG_CFA import Bayer_GRGR1
from CFA.RGBW_CFA import RGBW_CFA2
from add_noise import add_noise


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root,block_size, std,bool_noise,split='train'):
        super(Places2, self).__init__()
        self.std=std
        self.bool_noise=bool_noise
        if split == 'train':
            self.paths = glob('{:s}/train_large/**/*.png'.format(img_root),
                              recursive=True)
            self.block_size=block_size
        else:
            self.paths = sorted(glob('{:s}/{:s}_large/*'.format(img_root, split)))
            self.block_size=block_size

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        # size = (gt_img.size[1],gt_img.size[0]) #高、宽
        gt_img_rgb=gt_img.convert('RGB') #无需加噪音

        #加噪音
        # gt_img_rgb=gt_img.convert('RGB')/255 #加噪音先归一化到0--1
        # gt_img_rgb=add_noise(gt_img_rgb,mode="gaussian",var=0.01) #var表示噪音大小
        # x=np.array(gt_img_rgb).transpose(2,0,1)
        # gt_img_rgb=np.array(gt_img_rgb)
        # print('gt_img_rgb',gt_img_rgb)
        # print('gt_img_rgb.shape',gt_img_rgb.shape) # (1359, 2041, 3)
        # w_gt=gt_img_rgb[:,:,0]+gt_img_rgb[:,:,1]+gt_img_rgb[:,:,2]
        # print('w_gt.shape', w_gt.shape)#(1359, 2041)
        # w_gt=w_gt[:,:,None]
        # print('w_gt.shape',w_gt.shape) #(1359, 2041, 1)
        # rgbw_img=np.concatenate((gt_img_rgb,w_gt),axis=2)
        # print('rgbw_img',rgbw_img.transpose(2,0,1)[:,::3,::3])
        # self.x=transforms.ToTensor()
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()])#transforms.CenterCrop((self.block_size[1],self.block_size[2])),transforms.CenterCrop((self.block_size[1],self.block_size[2])),transforms.Resize(size), transforms.CenterCrop((512,512)), transforms.RandomCrop((128,128)),

        # gt_img_tensor = self.img_transform(gt_img.convert('RGB')) #torch.Size([3, 512, 512])
        gt_img_tensor = self.img_transform(gt_img_rgb)
        # z=self.x(rgbw_img)
        # print('gt_img_tensor.shape',gt_img_tensor.shape) #torch.Size([3, 256, 256])
        # save_image(gt_img_tensor,'../snapshots/default/images_rgbw/learn_rgbw/gt.png')

        # bayer_img,mask=Bayer_GRGR1(gt_img_tensor) #测试rgb

        w_gt_tensor=gt_img_tensor[0,:,:]+gt_img_tensor[1,:,:]+gt_img_tensor[2,:,:]
        w_gt_tensor=w_gt_tensor[None,:,:]
        image_tensor_rgbw=torch.cat((gt_img_tensor,w_gt_tensor),dim=0)
        # print('11111111',image_tensor_rgbw>1)
        # print('x',x)
        # print('image_tensor_rgbw',image_tensor_rgbw)
        # exit(0)
        # x=torch.clamp(image_tensor_rgbw,0,1)
        # print(z==x)
        # exit()
        # image_tensor_rgbw=torch.clamp(image_tensor_rgbw,0,1)
        image_tensor_rgbw=image_tensor_rgbw/torch.max(image_tensor_rgbw)
        # print('222222222',image_tensor_rgbw>1)
        # gt_img_rgbw=image_tensor_rgbw
        gt_img_rgbw=torch.clone(image_tensor_rgbw)
        # print('image_tensor_rgbw', image_tensor_rgbw)
        # print(gt_img_rgbw>1)
        # exit()
        # print('image_tensor_rgbw.shape',image_tensor_rgbw.shape) #torch.Size([4, 256, 256])
        #add noise
        if self.bool_noise==True:
            # print('添加高斯噪声std={}'.format(self.std))
            # noise=np.random.normal(0, self.std, image_tensor_rgbw.shape)
            # noise_rgbw=torch.from_numpy(noise)
            # image_tensor_rgbw=image_tensor_rgbw+noise_rgbw
            image_tensor_rgbw+=torch.from_numpy((np.random.normal(0, self.std, image_tensor_rgbw.shape)))
            image_tensor_rgbw=torch.clamp(image_tensor_rgbw,0,1)
        # print('11111',image_tensor_rgbw)
        
        
        # bayer_rgbw,mask=RGBW_CFA2(image_tensor_rgbw) #测试rgbw
        # print(gt_img_rgbw>1)
        return image_tensor_rgbw,gt_img_rgbw #RGBW 返回
        # return image_tensor_rgbw,gt_img_rgbw[0:3] #RGBW 返回 gt为三通道
        # return gt_img_tensor,gt_img_tensor #RGB返回
        # return bayer_img,gt_img_tensor #经过Bayer_rgrg cfa的RGB返回
        # return bayer_rgbw,image_tensor_rgbw #经过rgbw cfa的RGBw返回

    def __len__(self):
        return len(self.paths)