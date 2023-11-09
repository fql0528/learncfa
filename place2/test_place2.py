
import torch
from PIL import Image
from glob import glob
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image


class Places2_yuanchicun(torch.utils.data.Dataset):
    def __init__(self, img_root,std,bool_noise, split='train'):
        super(Places2_yuanchicun, self).__init__()
        self.std=std
        self.bool_noise=bool_noise
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.png'.format(img_root),
                              recursive=True)
        else:
            self.paths = sorted(glob('{:s}/{:s}_large/*'.format(img_root, split)))

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        # size = (gt_img.size[1],gt_img.size[0]) #高、宽
        gt_img_rgb=gt_img.convert('RGB')
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()])#transforms.RandomCrop((256,256)), transforms.CenterCrop((512,512)),transforms.Resize(size),

        gt_img_tensor = self.img_transform(gt_img_rgb) #torch.Size([3, 512, 512])
        w_gt_tensor=gt_img_tensor[0,:,:]+gt_img_tensor[1,:,:]+gt_img_tensor[2,:,:]
        w_gt_tensor=w_gt_tensor[None,:,:]
        image_tensor_rgbw=torch.cat((gt_img_tensor,w_gt_tensor),dim=0)
        # gt_img_rgbw=image_tensor_rgbw
        max_w=torch.max(image_tensor_rgbw)
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

        return image_tensor_rgbw, gt_img_rgbw,max_w

    def __len__(self):
        return len(self.paths)