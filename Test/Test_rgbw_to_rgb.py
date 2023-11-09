import argparse
import numpy as np
import torch
from place2.test_place2 import Places2_yuanchicun
from torch.utils.data import DataLoader
from src import mymodels,summary_utils,utils
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image
from psnr_ssim.PSNR_SSIM import calc_psnr_tensor_to_255_uint8
import time
import torch.nn as nn
from PIL import Image
import imageio
from  torchvision import utils as vutils
utils.seed(num=123,deterministic=False)
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    start_time=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../canon_data/Gehler_Shi568_Convert_Qiao')
    # parser.add_argument('--root', type=str, default='../canon_data/568_great_test')
    parser.add_argument('--test_epoch',type=str,default='new_noise0.04_best')
    parser.add_argument('--snapshot', type=str, default='../snapshots/default')
    parser.add_argument('--save_image_dir', type=str,
                        default='../image_result', help='保存图片路径')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--bool_noise',type=bool,default=True)
    parser.add_argument('--noise_std',type=float,default=0.04)
    parser.add_argument('-b', '--block_size',
                            help='delimited list input for block size in format(格式中块大小的分隔列表输入,只针对lrgbw的训练和测试) %,%,%',
                            default=[4,128,128])
    parser.add_argument('--cfa_size', help='cfa的尺寸大小,只针对lrgbw的训练和测试有效%,%,%',
                        default=[4,4])
    parser.add_argument('--w_zhi', type=float,default=0.001)
    parser.add_argument('--alpha', type=float,default=2.5)
    parser.add_argument('--gaussian_weight_size', type=int,default=7)
    parser.add_argument('--interp', type=str, choices=['none', 'bilinear', 'scatter','pconv','gaussian'],
                        default='gaussian') #原版required=True, default='pconv'
    parser.add_argument('--init', type=str, choices=['softmax_tau', 'gumbel_softmax_tau_1', 'gumbel_softmax_tau'], 
                        default='softmax_tau',
                    help='针对可学习RGBW CFA的形式,反向传播代理函数问题,'
                            'softmax_tau:带温度系数的softmax函数,gama=2.5e-5,可去函数里面调整,tau根据迭代次数进行变化,'
                            'gumbel_soft_tau_1:Gumbel_Softmax函数,tau为初始值1,'
                            'gumbel_softmax_tau:Gumbel_Softmax,tau可变,根据迭代次数变化')
    parser.add_argument('--num_workers', type=int, default=2) #原版1
    parser.add_argument('--decoder', type=str,
                        choices=['unet', 'mpr', 'dncnn', 'myunet','sgnet',
                                 'mysgnet','simdunet','rlfn','nafnet',
                                 'mysgnet1','rrdbnet','shufflemixer',
                                 'msanet','rdunet','unetres'],
                        default='mysgnet1')  #原版default='unet'
    parser.add_argument('--shutter', type=str, 
                        choices=['rgbw-Kodak RGBW',' gindelergbw','luorgbw',
                        'wangrgbw','cfzrgbw','nipsrgbw','binningrgbw','sonyrgbw',
                        'yamagamirgbw','kaizurgbw','hamiltonrgbw','hondargbw',
                        'randomrgbw','ourrgbw','bayerrgb'],default='kaizurgbw') #quad 原版required=True, default='lsvpe'  default='rgbw'
    args = parser.parse_args()
    snapshot='{:s}/ckpt/{:s}/{:s}/{:s}_epoch.pth'.format(args.snapshot, args.shutter,args.interp,args.test_epoch)
    datasets_test=Places2_yuanchicun(img_root=args.root,std=args.noise_std,bool_noise=args.bool_noise,split='test')
    print('len(datasets_test)',len(datasets_test))
    test_dataloader=DataLoader(datasets_test,
                        batch_size=1,
                        num_workers=args.num_workers,
                        shuffle=False,
                        )#pin_memory=True
    print('len(test_dataloader)',len(test_dataloader))
    shutter=mymodels.define_myshutter(args.shutter,args)
    decoder=mymodels.define_mydecoder(args.decoder,args)
    model=mymodels.define_mymodel(shutter,decoder,args,get_coded=False)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # print('model',model)
    # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        # print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型

    # """
    #该论文模型加载
    #考虑到多GPU训练模型加载的问题
    #https://blog.csdn.net/u013250861/article/details/126884553?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EAD_ESQUERY%7Eyljh-1-126884553-blog-90516365.235%5Ev27%5Epc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EAD_ESQUERY%7Eyljh-1-126884553-blog-90516365.235%5Ev27%5Epc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=2
    checkpoint=torch.load(snapshot)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    # """
    # utils.load_myckpt(args.snapshot,[('model', model)])  #PConv 加载模型  要与训练的保存模型相对应
    # print('args',args)
    model.to(device)
    model.eval()
    with torch.no_grad():
        psnr_rgb_list=[]
        ssim_rgb_list=[]
        step=0
        save_image_index=10
        for step,(model_input, gt,max_w) in enumerate(test_dataloader):
        # for (model_input, gt) in tqdm(test_dataloader):
            # print('model_input.shape',model_input.shape) #
            # print('model_input',model_input)
            step+=1
            # print('step',step)
            # print('gt.shape',gt.shape)#
            model_input = model_input.to(device)
            gt = gt.to(device)
            deblur ,cfa_img= model(model_input,train=False)
            #cfa_img为3通道RGB tensor直接可以保存，维度为[3,h,w]
            gt_rgb=gt[:,:3,:,:]
            
            # save_image(gt_rgb,'./gt.png')
            # psnr_rgb=0
            # for i in range(gt_rgb.shape[1]):
                # psnr_rgb+=summary_utils.get_psnr(deblur,gt_rgb)
            # psnr_rgb=psnr_rgb/3
            psnr_rgb=summary_utils.get_psnr(deblur,gt_rgb)
            # ssim_rgb=summary_utils.get_ssim(deblur,gt_rgb)
            
            #uint8计算方法
            # psnr_rgb=summary_utils.get_psnr_uint8(deblur,gt_rgb)

            # psnr=calc_psnr_tensor_to_255_uint8(deblur_rgb,gt_rgb)
            # print('psnr',psnr)

            print("第{:d}张图片PSNR：".format(step),psnr_rgb)
            psnr_rgb_list.append(psnr_rgb)
            # print("第{:d}张图片SSIM：".format(step),ssim_rgb)
            # ssim_rgb_list.append(ssim_rgb)

            # print('psnr_rgb',psnr_rgb)
            save_path='{:s}/{:s}/{:s}/{:d}.png'.format(args.save_image_dir,args.shutter,args.interp,save_image_index)
            # deblur_rgb=deblur_rgb*255
            max_w=max_w.to(device)
            # save_image(deblur*max_w, save_path) #保存重建图像
            # deblur_rgb=deblur_rgb/torch.max(deblur_rgb)
            
            # print(save_path)
            # exit()
            # save_image(deblur*max_w, save_path)#
            save_image_index += 10
            # vutils.save_image(deblur_rgb, save_path, normalize=True) #有一点暗
            # vutils.save_image(deblur_rgb, save_path)#非常暗

            """
            rgb_img=deblur_rgb[0].cpu().detach()
            rgb_img=np.array(rgb_img*255,dtype='uint8').transpose(1,2,0)
            rgb_img=Image.fromarray(rgb_img).convert("RGB")
            rgb_img.save(save_path)
            """

        print('psnr_rgb_list',psnr_rgb_list)

        print('psnr_rgb_mean',np.mean(psnr_rgb_list),'--len(psnr_rgb_list)',len(psnr_rgb_list) )
        # print('ssim_rgb_mean',np.mean(ssim_rgb_list),'--len(ssim_rgb_list)',len(ssim_rgb_list) )
        # print('len(psnr_rgb_list)',len(psnr_rgb_list))

    print("block_size:",args.block_size)
    print("shutter: ",args.shutter ) 
    print("interpolation: ",args.interp)
    print("decoder:",args.decoder)
    print("是否加噪声:",args.bool_noise)
    print("噪声等级std:",args.noise_std)
    end_time=time.time()
    print(f"程序运行时间：{end_time-start_time:.4f}s")
