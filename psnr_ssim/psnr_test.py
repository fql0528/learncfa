from  PSNR_SSIM import  calc_psnr
import  numpy as np
from PIL import Image


def get_psnr(pred, gt):
    return 10 * np.log10(1 / np.mean((pred - gt) ** 2))
gt='./0010.png'
img='./10.png'
img_an='./10_an.png'
psnr=calc_psnr(gt,img_an)
print(psnr)

img1 = Image.open(gt).convert('RGB')
img2 = Image.open(img).convert("RGB")
img1, img2 = np.array(img1)/255, np.array(img2)/255
img1,img2=img1.transpose(2,0,1),img2.transpose(2,0,1)
w1=img1[0,:,:]+img1[1,:,:]+img1[2,:,:]
w2=img2[0,:,:]+img2[1,:,:]+img2[2,:,:]
img1=img1/np.max(w1)
img2=img2/np.max(w2)
print(img1.shape)
psnr1=get_psnr(img1,img2)
print('psnr1',psnr1)