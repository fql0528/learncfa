import torch
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import make_grid
import math


def get_psnr(pred, gt):
    return 10 * torch.log10(1 / torch.mean((pred - gt) ** 2)).detach().cpu().numpy()

#uint8用这个函数
def get_psnr_uint8(pred, gt):
    # print(torch.round(torch.clamp(gt*255,0,255)))
    return 10 * torch.log10(255*255 / torch.mean((torch.round(torch.clamp(pred*255,0,255)) - torch.round(torch.clamp(gt*255,0,255))) ** 2)).detach().cpu().numpy()

def get_psnr_255(pred, gt):
    return psnr(pred, gt,data_range=255)

#来源于2018联合学习RGB CFA和去马赛克
def mse(predictions,targets):
    return np.sum(((predictions - targets) ** 2))/(predictions.shape[3]*predictions.shape[1]*predictions.shape[2])

def cpsnr(img1, img2):
    img1=img1.detach().cpu().numpy()*255
    img2=img2.detach().cpu().numpy()*255
    mse_tmp = mse(np.round(np.clip(img1,0,255)),np.round(np.clip(img2,0,255)))
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse_tmp)

def calc_psnr_tensor_to_255_uint8(img1,img2):
    #输入的img1 和img2均为0-1之间的tensor
    # img1,img2=np.array(img1*255.0,dtype='uint8'),np.array(img2*255.0,dtype='uint8')
    # img1,img2=np.asarray(img1*255,dtype='uint8'),np.asarray(img2*255,dtype='uint8')
    img1,img2=img1.cpu().detach(),img2.cpu().detach()
    img1,img2=np.asarray(img1*255),np.asarray(img2*255)
    img1,img2=np.round(img1,0),np.round(img2,0)
    img1,img2=np.array(np.clip(img1,0,255),dtype=np.uint8),np.array(np.clip(img2,0,255),dtype=np.uint8)
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score

def get_ssim(pred, gt):
    ssims = []
    for i in range(pred.shape[0]):
        pred_i = pred[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = gt[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        ssims.append(ssim(pred_i, gt_i, multichannel=True))
    return sum(ssims) / len(ssims)


def write_val_scalars(writer, names, values, total_steps):
    for name, val in zip(names, values):
        writer.add_scalar(f'val/{name}', np.mean(val), total_steps)


def write_summary(batch_size, writer, shutter_name, model, input, gt,
                  output, avg, total_steps, optim):
    coded= model.shutter(input) #原版
    # coded,mask = model.shutter(input)# Pconv

    cat_input = coded[:, 0, :, :].unsqueeze(0)
    for i in range(1, coded.shape[1]):
        cat_input = torch.cat((cat_input, coded[:, i, ...].unsqueeze(0)), dim=0)

    grid = make_grid(cat_input,
                     scale_each=True, nrow=1, normalize=False).cpu().detach().numpy()
    writer.add_image(f"sensor image", grid, total_steps)

    result_gt = torch.cat((avg, output.cpu(), gt.cpu()), dim=0)
    grid = make_grid(result_gt,
                     scale_each=True,
                     nrow=batch_size,
                     normalize=False).cpu().detach().numpy()
    writer.add_image(f"avg_result_gt", grid, total_steps)

    psnr = get_psnr(output, gt)
    ssim = get_ssim(output, gt)
    writer.add_scalar(f"train/psnr", psnr, total_steps)
    writer.add_scalar(f"train/ssim", ssim, total_steps)
    writer.add_scalar("learning_rate", optim.param_groups[0]['lr'], total_steps)

    if 'learn' in shutter_name:
        fig = plt.figure()
        plt.bar(model.shutter.counts.keys(), model.shutter.counts.values())
        plt.ylabel('counts')
        writer.add_figure(f'lengths_freq', fig, total_steps)

        shutter = model.shutter.lengths.detach().cpu()

        fig = plt.figure()
        plt.imshow(shutter)
        plt.colorbar()
        writer.add_figure(f'train/learned_length', fig, total_steps)

def write_summary_rgbw(batch_size, writer, shutter_name, model, input, gt,
                  output, avg, total_steps, optim):
    # coded= model.shutter(input) #原版
    coded,mask = model.shutter(input)# rgbw

    cat_input = coded[:, 0, :, :].unsqueeze(0)
    for i in range(1, coded.shape[1]):
        cat_input = torch.cat((cat_input, coded[:, i, ...].unsqueeze(0)), dim=0)

    grid = make_grid(cat_input,
                     scale_each=True, nrow=1, normalize=False).cpu().detach().numpy()
    writer.add_image(f"sensor image", grid, total_steps)
    # print('avg.shape',avg.shape)#torch.Size([1, 4, 512, 512])
    # print('output.shape',output.shape) # torch.Size([1, 4, 512, 512])
    # print('gt.shape',gt.shape)#torch.Size([1, 4, 512, 512])
    result_gt = torch.cat((avg, output.cpu(), gt.cpu()), dim=0)
    grid = make_grid(result_gt,
                     scale_each=True,
                     nrow=batch_size,
                     normalize=False).cpu().detach().numpy()
    writer.add_image(f"avg_result_gt", grid, total_steps)

    psnr = get_psnr(output, gt)
    ssim = get_ssim(output, gt)
    writer.add_scalar(f"train/psnr", psnr, total_steps)
    writer.add_scalar(f"train/ssim", ssim, total_steps)
    writer.add_scalar("learning_rate", optim.param_groups[0]['lr'], total_steps)

    if 'learn' in shutter_name:
        fig = plt.figure()
        plt.bar(model.shutter.counts.keys(), model.shutter.counts.values())
        plt.ylabel('counts')
        writer.add_figure(f'lengths_freq', fig, total_steps)

        shutter = model.shutter.lengths.detach().cpu()

        fig = plt.figure()
        plt.imshow(shutter)
        plt.colorbar()
        writer.add_figure(f'train/learned_length', fig, total_steps)
