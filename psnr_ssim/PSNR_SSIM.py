from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
#https://zhuanlan.zhihu.com/p/309892873

def calc_psnr(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).

    References
    -------
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    '''
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # x1=img1.transpose(2,0,1)
    # x2 = img2.transpose(2, 0, 1)
    # print('img1',x1)
    # print('img2',x2)
    # x=(img1/255).transpose(2,0,1)
    # print('x.shape',x.shape)
    # print('img1',x)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    # print(img1.shape,img2.shape)
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score

def calc_psnr_tensor(img1,img2):
    #输入的img1 和img2均为0-1之间的tensor
    img1,img2=np.asarray(img1),np.asarray(img2)
    # print('img1',img1)
    # print('img1.shape',img1.shape) #(1, 3, 1359, 2041)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=1)
    return psnr_score

def calc_psnr_tensor_to_255_uint8(img1,img2):
    #输入的img1 和img2均为0-1之间的tensor
    # img1,img2=np.array(img1*255.0,dtype='uint8'),np.array(img2*255.0,dtype='uint8')
    # img1,img2=np.asarray(img1*255,dtype='uint8'),np.asarray(img2*255,dtype='uint8')
    img1,img2=img1.cpu().detach(),img2.cpu().detach()
    img1,img2=np.asarray(img1*255),np.asarray(img2*255)
    img1,img2=np.round(img1,0),np.round(img2,0)
    img1,img2=np.array(np.clip(img1,0,255),dtype=np.uint8),np.array(np.clip(img2,0,255),dtype=np.uint8)
    
    # print('img1',img1)
    # print('img2',img2)
    # gt_img = Image.open('../canon_dataset/568_great/test_large/0010.png')
    # gt_img = Image.open('../image_result/PConv/result_output_comp_1.png')
    # gt_img_array=np.array(gt_img).transpose(2,0,1)  
    # print(img2==gt_img_array) #全为true

    # img1,img2=np.round(img1),np.round(img2)
    # print('img1_1',img1)
    # print('img2_2',img2)
    # img1,img2=np.uint8(img1),np.uint8(img2) #不要用这个
    # print('img1',img1)
    # print('img2',img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    # print(img1.shape,img2.shape)
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score