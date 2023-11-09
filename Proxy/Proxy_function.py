import sys

import shutters.shutter_utils as shutils
import torch
import torch.nn.functional as F
import math
from Soft_max.My_Gumbel_Softmax import my_gumbel_softmax

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
T=1


def Proxy_function(cfa_image,cfa,init,image,w,mask,alpha):
    # cfa_image=torch.zeros(size=(image.shape),device=device)
    # cfa=torch.zeros(size=(image.shape[1:]),device=device)
    if init=='softmax_tau':
        # print('alpha',alpha)
        # exit()
        #带温度系数的softmax函数
        # print('代理函数：带温度系数的softmax函数')
        x = shutils.produce_rgbw_cfa(w,alpha_0=alpha)
        # x.register_hook(print)
        # print('x.shape',x.shape)
        # print('x',x)
        # exit()
        for i in range(4):
            cfa[i]=torch.sum(x[:,i]*mask[:,i],dim=0,keepdim=True)
            cfa_image[:,i]=cfa[i]*image[:,i]
        return cfa_image,cfa
    elif init=='gumbel_softmax_tau_1':
        #Gumbel_Softmax 
        # print('代理函数：Gumbel_Softmax函数,tau=1')
        # tau=1.0
        # x=F.gumbel_softmax(logits=w,hard=True,tau=tau,dim=1)
        x=my_gumbel_softmax(logits=w, hard=True, tau=alpha, dim=1) #未加gumbel分布
        # print('x',x)
        # exit()
        for i in range(4):
            cfa[i]=torch.sum(x[:,i]*mask[:,i],dim=0,keepdim=True)
            cfa_image[:,i]=cfa[i]*image[:,i]
        return cfa_image,cfa
    elif init=='gumbel_softmax_tau':
        # Gumbel_SoftMax
        print("代理函数：Gumbel_Softmax函数,tau可变")
        t0=10 #初始温度
        # T=1 #迭代次数
        global T #迭代次数
        tau = t0 / (1 + math.log(T))
        x=F.gumbel_softmax(logits=w,hard=True,tau=tau,dim=1)
        for i in range(4):
            cfa[i]=torch.sum(x[:,i]*mask[:,i],dim=0,keepdim=True)
            cfa_image[:,i]=cfa[i]*image[:,i]
        T+=1
        # print('T',T)
        return cfa_image,cfa
    else:
        print("选择了未知代理函数,程序error")
        sys.exit(-1)