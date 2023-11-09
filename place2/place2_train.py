import torch
from PIL import Image
from glob import glob
import numpy as np
from torchvision import transforms


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, split='train'):
        super(Places2, self).__init__()
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.png'.format(img_root),
                              recursive=True)
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index]).convert('RGB') #
        size = (gt_img.size[1], gt_img.size[0])  # 高、宽
        # print('size',size) #(186, 317)
        gt_img_array=np.array(gt_img)
        # print('gt_img_array.shape',gt_img_array.shape)# (186, 317, 3)
        gt_img_tensor=torch.Tensor(gt_img_array)
        # print('gt_img_tensor.shape',gt_img_tensor.shape) #torch.Size([186, 317, 3])
        gt_img_tensor=gt_img_tensor.permute(2,0,1)
        # print('gt_img_tensor.shape', gt_img_tensor.shape) #torch.Size([3, 186, 317])

        self.img_transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.ToTensor()])  # transforms.CenterCrop((512,512)), transforms.RandomCrop(size=(256,384))

        # print('gt_img.size',gt_img.size) #
        gt_img_tensor = self.img_transform(gt_img.convert('RGB'))  # torch.Size([3, 512, 512])
        # gt_raw_tensor=self.img_transform(gt_img.convert("L")) #torch.Size([1, 512, 512])

        # print('gt_raw_tensor.shape',gt_raw_tensor.shape) #torch.Size([1, 512, 512])
        # print('gt_img_tensor.shape',gt_img_tensor.shape) #torch.Size([3, 512, 512])
        # image_array=np.array(gt_img_tensor)
        # print('image_array',image_array)
        # print("image_array.shape",image_array.shape) #(3, 512, 512)
        # w_gt=image_array[0,:,:]+image_array[1,:,:]+image_array[2,:,:]
        w_gt_tensor = gt_img_tensor[0, :, :] + gt_img_tensor[1, :, :] + gt_img_tensor[2, :, :]
        w_gt_tensor = w_gt_tensor[None, :, :]
        # print('w_gt_tensor',w_gt_tensor*255)
        # print('w_gt_tensor.shape',w_gt_tensor.shape) #torch.Size([1, 512, 512])
        image_tensor_rgbw = torch.cat((gt_img_tensor, w_gt_tensor), dim=0)
        # print('image_tensor_rgbw',image_tensor_rgbw)
        # print('image_tensor_rgbw.shape',image_tensor_rgbw.shape) # torch.Size([4, 186, 317])

        # image_array_rgbw=np.concatenate((image_array,w_gt),axis=0)
        # print('image_array_rgbw',image_array_rgbw)
        # print('image_array_rgbw.shape',image_array_rgbw.shape) # (4, 512, 512)
        # bayer_rgbw,mask=RGBW_CFA(image_array) #Bayer1（）输出为【通道、高、宽】的array
        # print('###########################################')
        # print('mask.shape',mask.shape) #(4, 1359, 2041)
        # print('bayer_rgbw.shape',bayer_rgbw.shape) #(4, 1359, 2041)
        # bayer_rgbw_tensor=torch.Tensor(bayer_rgbw)
        # mask_tensor=torch.Tensor(mask)

        # print('bayer_rgbw_tensor',bayer_rgbw_tensor)
        # print('bayer_rgbw_tensor.shape',bayer_rgbw_tensor.shape) #torch.Size([4, 1359, 2041])
        # avg_tensor=torch.mean(bayer_rgbw_tensor, dim=0, keepdim=True) #

        # avg_tensor=torch.mean(image_tensor_rgbw,dim=0,keepdim=True) #原版 torch.Size([1, 512, 512])
        # print('avg_tesnor.shape',avg_tensor.shape)#torch.Size([1, 512, 512])
        avg_tensor = image_tensor_rgbw

        # print('image_tensor_rgbw',image_tensor_rgbw)
        # x=image_tensor_rgbw[:3,:,:]
        # save_image(x,'../image_result/rgbw/gt_256x384.png')
        # return avg_tensor,image_tensor_rgbw, gt_raw_tensor
        # print('gt_img_tensor',gt_img_tensor)
        return gt_img_tensor,image_tensor_rgbw

    def __len__(self):
        return len(self.paths)