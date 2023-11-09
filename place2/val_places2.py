
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from CFA.RGRG_CFA import Bayer_GRGR1
from CFA.RGBW_CFA import RGBW_CFA2

class Places2_yuanchicun(torch.utils.data.Dataset):
    def __init__(self, img_root,std,bool_noise, split='train'):
        super(Places2_yuanchicun, self).__init__()
        self.std=std
        self.bool_noise=bool_noise
        if split == 'train':
            self.paths = glob('{:s}/train_large/**/*.png'.format(img_root),
                              recursive=True)
        else:
            self.paths = sorted(glob('{:s}/{:s}_large/*'.format(img_root, split)))

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        
        # size = (gt_img.size[1],gt_img.size[0]) #高、宽
        gt_img_rgb=gt_img.convert('RGB') #无需加噪音

        #加噪音
        # gt_img_rgb=gt_img.convert('RGB')/255 #加噪音先归一化到0--1
        # gt_img_rgb=add_noise(gt_img_rgb,mode="gaussian",var=0.01) #var表示噪音大小

        self.img_transform = transforms.Compose(
            [transforms.ToTensor()])#transforms.Resize(size), transforms.CenterCrop((512,512)), ,transforms.CenterCrop((120,120)),
        # gt_img_tensor = self.img_transform(gt_img.convert('RGB')) #torch.Size([3, 512, 512])
        gt_img_tensor = self.img_transform(gt_img_rgb)
        # bayer_img,mask=Bayer_GRGR1(gt_img_tensor)

        w_gt_tensor=gt_img_tensor[0,:,:]+gt_img_tensor[1,:,:]+gt_img_tensor[2,:,:]
        w_gt_tensor=w_gt_tensor[None,:,:]
        image_tensor_rgbw=torch.cat((gt_img_tensor,w_gt_tensor),dim=0)
        image_tensor_rgbw = image_tensor_rgbw / torch.max(image_tensor_rgbw)
        # image_tensor_rgbw = torch.clamp(image_tensor_rgbw, 0, 1)
        # gt_img_rgbw=image_tensor_rgbw
        gt_img_rgbw=torch.clone(image_tensor_rgbw)



        #add noise
        if self.bool_noise==True:
            # print('添加高斯噪声std={}'.format(self.std))
            # noise=np.random.normal(0, self.std, image_tensor_rgbw)
            # noise_rgbw=torch.from_numpy(noise)
            # image_tensor_rgbw=image_tensor_rgbw+noise_rgbw
            image_tensor_rgbw += torch.from_numpy((np.random.normal(0, self.std, image_tensor_rgbw.shape)))
            image_tensor_rgbw = torch.clamp(image_tensor_rgbw, 0, 1)
        # bayer_rgbw,mask=RGBW_CFA2(image_tensor_rgbw) #测试rgbw

        return image_tensor_rgbw,gt_img_rgbw #RGBW 返回
        # return gt_img_tensor,gt_img_tensor #RGB返回
        # return bayer_img,gt_img_tensor #经过Bayer_rgrg cfa的RGB返回
        # return bayer_rgbw,image_tensor_rgbw #经过rgbw cfa的RGBw返回

    def __len__(self):
        return len(self.paths)