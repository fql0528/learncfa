from glob import glob
from PIL import Image
import os
import cv2
import math

'''
把大图像切成指定大小的图像块，不够的最后一个patch，倒着从最后往前截取一个patch的大小。
https://blog.csdn.net/Crystal_remember/article/details/128988235?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128988235-blog-126139559.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128988235-blog-126139559.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=6
'''

def corp_batch(img_root,save_root,split,corp_h,corp_w):
    path=glob('{:s}/{:s}_large/**/*.png'.format(img_root,split),
                              recursive=True)
    # print(len(path))
    # exit(0)
    for i  in range(len(path)):
        p,n = os.path.split(path[i])
        pro,ext = os.path.splitext(n)
        img=cv2.imread(path[i])
        # print(img.shape)
        (h, w, c) = img.shape
        h_n = math.ceil(h / corp_h)  #裁剪的高  480修改为想裁剪的大小
        w_n = math.ceil(w / corp_w)  #裁剪的宽  480修改为想裁剪的大小
        for i in range(h_n):
            if i < h_n - 1:
                for j in range(w_n):
                    if j < w_n - 1:
                        img_patch = img[i * corp_h:(i + 1) * corp_h, j * corp_w:(j + 1) * corp_w, :]
                        img_pathname = os.path.join('{:s}'.format(save_root),
                                                    pro +'_'+ str(i) + '_' + str(j) + '.png')
                        cv2.imwrite(img_pathname, img_patch)
    
                    else:
                        img_patch = img[i * corp_h:(i + 1) * corp_h, (w - corp_w):, :]
                        img_pathname = os.path.join('{:s}'.format(save_root),
                                                    pro +'_'+ str(i) + '_' + str(j) + '.png')
                        cv2.imwrite(img_pathname, img_patch)
    
            else:
                for j in range(w_n):
                    if j < w_n - 1:
                        img_patch = img[(h - corp_h):, j * corp_w:(j + 1) * corp_w, :]
                        img_pathname = os.path.join('{:s}'.format(save_root),
                                                    pro +'_'+ str(i) + '_' + str(j) + '.png')
                        cv2.imwrite(img_pathname, img_patch)
    
                    else:
                        img_patch = img[(h - corp_h):, (w - corp_w):, :]
                        img_pathname = os.path.join('{:s}'.format(save_root),
                                                    pro +'_'+ str(i) + '_' + str(j) + '.png')
                        cv2.imwrite(img_pathname, img_patch)
    return path



# img_root='./568_great_test'
# save_root='./568_great_test/data_large/'

img_root='./Gehler_Shi568_Convert_Qiao'
# save_root='./Gehler_Shi568_Convert_Qiao/train_batch_large/'
save_root='./Gehler_Shi568_Convert_Qiao/test_batch_128_large/'
corp_batch(img_root=img_root,split='test',save_root=save_root,corp_h=128,corp_w=128)

'''
print(path)
print(len(path))
gt_img = Image.open(path[0])
gt_img_rgb=gt_img.convert('RGB')
print(gt_img_rgb)
gt_img_tensor=gt_img_rgb
p,n = os.path.split(path[1])
pro,ext = os.path.splitext(n)
print('p',p)
print('n',n)
print('pro',pro)
print('ext',ext)



# paths=glob('../canon_data/568_great_test/test_large/*')
# print(paths)
'''