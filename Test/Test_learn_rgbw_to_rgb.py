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

if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# utils.seed(64)
utils.seed(num=123,deterministic=True)
if __name__ == '__main__':
    start_time=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../canon_data/Gehler_Shi568_Convert_Qiao')
    # parser.add_argument('--root', type=str, default='../canon_data/568_great_test')
    parser.add_argument('--test_epoch',type=str,default='Big94_best')
    parser.add_argument('--m',type=int,help='表示要读取的第几行数据',default=1)
    parser.add_argument('--snapshot', type=str, 
        default='../snapshots/default') #记得去init统一代理函数，若保存图片记得统一地址
    parser.add_argument('--save_image_dir', type=str, default='../image_result', help='保存图片路径')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--bool_noise',type=bool,default=True)
    parser.add_argument('--noise_std',type=float,default=0.01)
    parser.add_argument('-b', '--block_size',
                            help='delimited list input for block size in format(格式中块大小的分隔列表输入) %,%,%',
                            default=[4,256,256])
    parser.add_argument('--cfa_size', help='cfa的尺寸大小,只针对lrgbw有效%,%,%',
                        default=[4,4])
    parser.add_argument('--w_zhi', type=float,default=0.001)
    parser.add_argument('--alpha', type=float,default=2.0)
    parser.add_argument('--gaussian_weight_size', type=int,default=5)
    parser.add_argument('--interp', type=str, choices=['none', 'bilinear', 'scatter','pconv','gaussian'],
                        default='gaussian') #原版required=True, default='pconv'
    parser.add_argument('--init', type=str, choices=['softmax_tau', 'gumbel_softmax_tau_1', 'gumbel_softmax_tau'], 
                        default='softmax_tau',
                    help='针对可学习RGBW CFA的形式,反向传播代理函数问题,'
                            'softmax_tau:带温度系数的softmax函数,gama=2.5e-5,可去函数里面调整,tau根据迭代次数进行变化,'
                            'gumbel_soft_tau_1:Gumbel_Softmax函数,tau为初始值1,'
                            'gumbel_softmax_tau:Gumbel_Softmax,tau可变,根据迭代次数变化')
    parser.add_argument('--num_workers', type=int, default=1) #原版1
    parser.add_argument('--decoder', type=str,
                        choices=['unet', 'mpr', 'dncnn','myunet','sgnet'],
                        default='mysgnet1') #原版default='unet'
    parser.add_argument('--shutter', type=str, default='lrgbw') #quad 原版required=True, default='lsvpe'  default='rgbw'
    args = parser.parse_args()
    read_root_txt='{:s}/ckpt/{:s}/{:s}/set_canshu.txt'.format(args.snapshot, args.shutter,args.interp)
    fo_main=open(read_root_txt,'r')
    read_lines=fo_main.readlines()
    print('read_lines',read_lines)
    print('len(read_lines)',len(read_lines))
    # for m  in tqdm(range(1,len(read_lines))):
    print("要读取的数据为第{}行".format(args.m))
    m=args.m #表示要读取的第一行数据
    cfa_size=[]
    parameter=read_lines[m].strip('\n').split(',')
    args.alpha=float(parameter[3])
    args.w_zhi=float(parameter[4])
    args.gaussian_weight_size=int(parameter[5])
    cfa_sz=parameter[7].split('x')
    cfa_size.append(int(cfa_sz[0]))
    cfa_size.append(int(cfa_sz[1]))
    args.cfa_size=cfa_size
    # print(args.cfa_size)
    # exit()
    snapshot='{:s}/ckpt/{:s}/{:s}/{:s}/{:s}_epoch.pth'.format(args.snapshot, args.shutter,args.interp,args.init,args.test_epoch)
    print('snapshot',snapshot)
    datasets_test=Places2_yuanchicun(img_root=args.root, std=args.noise_std,bool_noise=args.bool_noise,split='test')
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
            # print(deblur<0)
            save_image(cfa_img[:3,:args.cfa_size[0],:args.cfa_size[1]],'cfa_last.png')
            # print(cfa_img[:3,:args.cfa_size[0],:args.cfa_size[1]])
            # exit()
            #cfa_img为3通道RGB tensor直接可以保存，维度为[3,h,w]
            gt_rgb=gt[:,:3,:,:]
            psnr_rgb=summary_utils.get_psnr(deblur,gt_rgb)
            # psnr=calc_psnr_tensor_to_255_uint8(deblur_rgb,gt_rgb)
            # print('psnr',psnr)

            print("第{:d}张图片PSNR：".format(step),psnr_rgb)
            psnr_rgb_list.append(psnr_rgb)

            # print('psnr_rgb',psnr_rgb)

            # 保存图片
            save_path='{:s}/{:s}/{:s}/{:s}/{:d}.png'.format(args.save_image_dir,args.shutter,args.interp,args.init,save_image_index)
            # deblur_rgb=deblur_rgb*255
            # save_image(deblur_rgb, save_path, normalize=True)
            max_w=max_w.to(device)
            # save_image(model_input[:,:3,:,:]*max_w,save_path)
            # save_image(deblur*max_w,save_path)
            save_image_index +=10



            # save_dir='{}111.png'.format(args.save_image_dir)
            # save_image(deblur_rgb,save_dir)
            # psnr=calc_psnr_tensor_to_255_uint8(gt_rgb,deblur_rgb)
            # print('psnr',psnr)
            # psnr_rgb=summary_utils.get_psnr(deblur_rgb,gt_rgb)
            # print('psnr_rgb',psnr_rgb)
            # p = summary_utils.get_psnr(deblur, gt)
            # print('p',p)
            # s = summary_utils.get_ssim(deblur, gt)
        print('psnr_rgb_list',psnr_rgb_list)

        print('psnr_rgb_mean',np.mean(psnr_rgb_list),'--len(psnr_rgb_list)',len(psnr_rgb_list) )
        # print('len(psnr_rgb_list)',len(psnr_rgb_list))

    print("block_size:",args.block_size)
    print("cfa_size:",args.cfa_size)
    print("shutter: ",args.shutter ) 
    print("interpolation: ",args.interp)
    print("decoder:",args.decoder)
    print('代理函数：',args.init)
    print('固定cfa初始化值为：',args.w_zhi)
    print("alpha: ",args.alpha)
    print("gaussian_weight_size: ",args.gaussian_weight_size)
    print("是否加噪声:",args.bool_noise)
    print("噪声等级std:",args.noise_std)
    end_time=time.time()
    print(f"程序运行时间：{end_time-start_time:.4f}s")
