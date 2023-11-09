import torch, os, lpips
import numpy as np
import random
import datetime
import pandas as pd

import torch.nn as nn
import nets.losses as losses


def seed(num,deterministic=False):
    random.seed(seed)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def modify_args(args):
    if args.decoder == 'unet' and (args.loss != 'l2_lpips' and args.loss != 'l2'):
        raise Exception('unet + loss do not match')
    if 'learn_all' in args.shutter and (args.interp != 'scatter' and args.interp != 'none'):
        raise Exception('learn all should use scatter or none')

    args.date_resume = args.resume
    if args.interp == 'none':
        args.interp = None
    # define folders 定义的文件夹
    if args.date_resume != '00-00-00':
        date = f'{args.date_resume}'
    else:
        date = datetime.date.today().strftime('%y-%m-%d')

    if args.test:
        exp_name = f'{args.log_root}/test'
        args.steps_til_summary = 10
    else:
        dir_name = f'{args.log_root}/{date}/{date}-{args.decoder}'
        print('dir_name',dir_name) #../logs/22-11-04/22-11-04-unet
        os.makedirs(dir_name, exist_ok=True)

        printed_args = get_exp_name(args) #block_size=[8, 512, 512]_scale=0_reg=100.0_k=3_p=1.0_mlr=0.0002_batch_size=2_steps_til_summary=8000_interp=scatter_init=quad_loss=l2_lpips_dec=unet_shut=lsvpe_sched=reduce_
        if args.exp_name != '':
            exp_name = f'{dir_name}/{args.exp_name}'
        else:
            exp_name = f'{dir_name}/{printed_args}'
        print('exp_name',exp_name)
        if args.date_resume != '00-00-00' and not os.path.exists(exp_name):
            raise ValueError('This directory does not exist :-(')
    return args, exp_name


def save_best_metrics(args, total_steps, epoch, val_psnrs, val_ssims, val_lpips, checkpoints_dir):
    single_column_names = ['Model', 'Shutter', 'Total Steps', 'Epoch', 'PSNR', 'SSIM', 'LPIPS']
    df = pd.DataFrame(columns=single_column_names)
    series = pd.Series([args.decoder,
                        args.shutter,
                        total_steps,
                        epoch,
                        round(np.mean(val_psnrs), 3),
                        round(np.mean(val_ssims), 3),
                        round(np.mean(val_lpips), 3)],
                       index=df.columns)
    df = df.append(series, ignore_index=True)

    file_name = f'{checkpoints_dir}/val_results.csv'
    # overwrite the file and just save the best recent
    df.to_csv(file_name, header=single_column_names)


def define_loss(args):
    loss_dict = {'mpr': losses.MPRNetLoss(),
                #  'l2_lpips': losses.L2LPIPSRegLoss(args.reg), #原版
                 'l1': nn.L1Loss(),
                 'l2': nn.MSELoss(),
                 'twol2':losses.TwoL2Loss()}
    return loss_dict[args.loss]


def make_model_dir(dir_name, test=False, exp_name=''):
    if test and exp_name != '':
        version_num = 0
        model_dir = f'{dir_name}/{exp_name}'
    else:
        version_num = find_version_number(dir_name) #v几的最后一个数字加1
        model_dir = f'{dir_name}/v_{version_num}'

    print('model_dir',model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, version_num


def define_schedule(optim):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                      factor=0.8,
                                                      patience=10,
                                                      threshold=1e-3,
                                                      threshold_mode='rel',
                                                      cooldown=0,
                                                      min_lr=5e-6,
                                                      eps=1e-08,
                                                      verbose=False)


def define_optim(model, args):
    # print(args.mlr) ## 0.0002
    # if 'lsvpe' or 'lrgbw' in args.shutter:  #注意 or 和in一块使用，in 的运算符顺序优于or，此结果为lsvpe，达不到想要的结果
    #     print('lrgbw' in args.shutter) #false
    #     print('lrgbw' in args.shutter)#false
    #     print('lsvpe' or 'lrgbw' in args.shutter) #lsvpe
    #     print('args.shutter',args.shutter) #rgb
    if 'lrgbw' in args.shutter:
        print("使用了AdamW，两者不同的学习率")
        optim = torch.optim.AdamW([{'params': model.shutter.parameters(), 'lr': args.slr},
                                   {'params': model.decoder.parameters(), 'lr': args.mlr}], lr=args.mlr, eps=1e-2)
    else:
        optim = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.mlr}], lr=args.mlr)
        # optim=torch.optim.Adam([{'params': model.parameters(), 'lr': args.mlr}], lr=args.mlr)
    return optim

def mydefine_optim(model, args):
    # print(args.mlr) ## 0.0002
    # if 'lsvpe' or 'lrgbw' in args.shutter:  #注意 or 和in一块使用，in 的运算符顺序优于or，此结果为lsvpe，达不到想要的结果
    #     print('lrgbw' in args.shutter) #false
    #     print('lrgbw' in args.shutter)#false
    #     print('lsvpe' or 'lrgbw' in args.shutter) #lsvpe
    #     print('args.shutter',args.shutter) #rgb
    if 'lrgbw' in args.shutter:
        print("使用了AdamW，两者不同的学习率")
        optim = torch.optim.AdamW([{'params': model.shutter.parameters(), 'lr': args.slr},
                                   {'params': model.decoder.parameters(), 'lr': args.mlr}],lr=args.mlr)
    else:
        optim = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.mlr}], lr=args.mlr)
        # optim=torch.optim.Adam([{'params': model.parameters(), 'lr': args.mlr}], lr=args.mlr)
    return optim


def save_chkpt(model, optim, checkpoints_dir, epoch=0, val_psnrs=None, final=False, best=False):
    if best:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()},
                   os.path.join(checkpoints_dir, 'model_best.pth'))
        return
    if final:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()},
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        return
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()},
                   os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

        if val_psnrs is not None:
            np.savetxt(os.path.join(checkpoints_dir, 'train_psnrs_epoch_%04d.txt' % epoch),
                       np.array(val_psnrs))
        return
        
def save_mychkpt(model, optim, checkpoints_dir):
    torch.save({'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict()},
                checkpoints_dir)
    """
    if torch.cuda.device_count() > 1:
        torch.save({'model_state_dict': model.module.state_dict(),
                'optim_state_dict': optim.module.state_dict()},
                checkpoints_dir)
    else:
        torch.save({'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict()},
                checkpoints_dir)
    """

def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict

def save_myckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

def load_myckpt(ckpt_name, models, optimizers=None):
    # ckpt_dict = torch.load(ckpt_name)  #原版
    ckpt_dict = torch.load(ckpt_name) #训练用cuda跑的，但是测试的时候加载模型如果用cup的话需要加上map_location=‘cpu’
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


def convert(img, dim=1):
    if dim == 1:
        return img.squeeze(0).squeeze(0).detach().cpu().numpy()
    if dim == 3:
        return img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


### functions for keeping train script clean  保持训练脚本干净的函数####
def make_subdirs(model_dir, make_dirs=True):
    summaries_dir = f'{model_dir}/summaries'
    checkpoints_dir = f'{model_dir}/checkpoints'
    if make_dirs:
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
    return summaries_dir, checkpoints_dir


def find_version_number(path):
    if not os.path.isdir(path):
        return 0
    fnames = sorted(os.listdir(path))
    # print('fnames',fnames) #['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8']
    latest = fnames[-1]
    #Python rsplit() 方法通过指定分隔符对字符串进行分割并返回一个列表，
    # 默认分隔符为所有空字符，包括空格、换行(\n)、制表符(\t)等。
    # 类似于 split() 方法，只不过是从字符串最后面开始分割。
    #https://blog.csdn.net/ZauberC/article/details/126108436
    latest = latest.rsplit('_', 1)[-1]
    return int(latest) + 1


def get_exp_name(args):
    ''' Make folder name readable  使文件夹名称可读'''
    printedargs = ''
    forbidden = ['data_root', 'log_root', 'test',
                 'remote', 'max_epochs', 'num_workers',
                 'epochs_til_checkpoint', 'date_resume',
                 'steps_til_ckpt', 'restart', 'slr', 'resume',
                  'gt', 'local', 'exp_name',]
    for k, v in vars(args).items():
        # print('vars(args).items()',vars(args).items()) #输出为字典，为args的名称和对应的值
        # print('k',k)
        # print('v',v)
        if k not in forbidden:
            print(f'{k} = {v}')
            if k == 'sched' and v == 'no_sched':
                continue
            if k == 'decoder':
                k = 'dec'
            if k == 'shutter':
                k = 'shut'
            printedargs += f'{k}={v}_'
            # print('printedargs',printedargs)
    # print('printedargs',printedargs) #block_size=[8, 512, 512]_scale=0_reg=100.0_k=3_p=1.0_mlr=0.0002_batch_size=2_steps_til_summary=8000_interp=scatter_init=quad_loss=l2_lpips_dec=unet_shut=lsvpe_sched=reduce_
    return printedargs


def augmentData(imgs):
    aug = random.randint(0, 8)
    num = len(imgs)
    # print('num',num) #3
    # Data Augmentations 数据增强
    if aug == 1:
        for i in range(num):
            imgs[i] = imgs[i].flip(-2)
    elif aug == 2:
        for i in range(num):
            imgs[i] = imgs[i].flip(-1)
    elif aug == 3:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1))
    elif aug == 4:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1), k=2)
    elif aug == 5:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1), k=3)
    elif aug == 6:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i].flip(-2), dims=(-2, -1))
    elif aug == 7:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i].flip(-1), dims=(-2, -1))
    return imgs
