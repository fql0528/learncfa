from torch.utils.data import DataLoader
from src.dataio import NFS_Video

REMOTE_CUSTOM_PATH = '/media/data6/cindy/custom_data'
# LOCAL_CUSTOM_PATH = '/home/cindy/PycharmProjects/custom_data'#原版
LOCAL_CUSTOM_PATH = '../custom_data'

block_sz = 512


def loadTrainingDataset(args, color=False, test=False):
    global REMOTE_CUSTOM_PATH #远程自定义路径
    if args.ares:
        REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data' #远程自定义路径
    # print('args.local',args.local) #True
    if not args.local:
        args.data_root = f'{REMOTE_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    else:
        args.data_root = f'{LOCAL_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    # print('args.data_root',args.data_root) #../custom_data/nfs_block_rgb_512_8f
    if args.test:  # use a smaller dataset when you're testing 测试时使用较小的数据集
        split = 'test'
    else:
        split = 'train'
    train_dataset = NFS_Video(log_root=args.data_root,
                                  block_size=args.block_size,
                                  gt_index=args.gt,
                                  color=color,
                                  split=split,
                                  test=test)
    if test:
        return DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False)
    return DataLoader(train_dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=True,
                      ) #原版有 pin_memory=True


def loadValDataset(args, color=False):
    global REMOTE_CUSTOM_PATH
    if args.ares:
        REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data'
    if not args.local:
        args.data_root = f'{REMOTE_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    else:
        args.data_root = f'{LOCAL_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    # print('args.test',args.test) #False
    if args.test:
        return None
    else:
        split = 'test'
   
    val_dataset = NFS_Video(log_root=args.data_root,
                                block_size=args.block_size,
                                gt_index=args.gt,
                                color=color,
                                split=split)

    return DataLoader(val_dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=False,
                      ) #原版有pin_memory=True
