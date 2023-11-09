import torch
from torch.autograd import Function
import torch.nn.functional  as F
import math
import random
import numpy as np
torch.manual_seed(123) 
torch.set_printoptions(threshold=8)
# device='cuda:0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t=1
my_alpha=None

''' Function for the binarized hadamard product between weights and inputs'''
def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2

def return_alpha(alpha_1):
    global my_alpha
    my_alpha=alpha_1
    

class BinarizeHadamardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        weight_b = where(weight>=0, 1, 0) # binarize weights
        output = input * weight_b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        weight_b = where(weight>=0, 1, 0) # binarize weights
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight_b
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output * input

        return grad_input, grad_weight


binarize_hadamard = BinarizeHadamardFunction.apply

class SignIncludingZero(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input >= 0
        return output.float()

    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -1, 1)
        return grad_input


class SignExcludingZero(Function):
    @staticmethod
    def forward(ctx, input):
        # print('ctx',ctx) #<torch.autograd.function.SignExcludingZeroBackward object at 0x000001760A210748>
        # print('ctx.shape',ctx.shape)
        # print('input',input)
        # print('input.shape',input.shape) #torch.Size([8, 512, 512])
        ctx.save_for_backward(input) #该ctx.save_for_backward方法用于存储在forward()此期间生成的值，
        #稍后将在执行时需要此值backward()。
        #可以backward()在ctx.saved_tensors属性期间访问保存的值
        output = input > 0
        # print('output',output) #里面全为bool类型，true或者false torch.Size([8, 512, 512])
        return output.float()

    def backward(ctx, grad_output):
        # print('grad_output',grad_output)
        # print('grad_output.shape',grad_output.shape) #torch.Size([8, 512, 512])
        grad_input = torch.clamp(grad_output, -1, 1)
        # print('grad_input',grad_input)
        return grad_input

class SignLRGBWZero(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) #该ctx.save_for_backward方法用于存储在forward()此期间生成的值，
        #稍后将在执行时需要此值backward()。
        #可以backward()在ctx.saved_tensors属性期间访问保存的值
        # output=torch.where(input==torch.amax(input,axis=0,keepdim=True),1.0,0.0)  #按维度比较最大值置为1其余置为0
        # print('input.shape',input.shape) #torch.Size([64, 4, 1, 1])
        output=torch.where(input==torch.amax(input,axis=1,keepdim=True),1.0,0.0)  #按维度比较最大值置为1其余置为0
        return output

    def backward(ctx, grad_output):
        # print('grad_output[0,:,:,:]',grad_output[0,:,:,:])
        # print('grad_output.shape',grad_output.shape) # torch.Size([64, 4, 1, 1])
        #带温度系数的softmax函数
        # gama=2.5e-5 #可设置值
        # t=1 #迭代次数
        # global t #迭代次数
        # alpha=1+(gama*t)**2

        # random.uniform(a, b): 返回随机生成的一个浮点数，范围在[a, b)之间# 
        # print('grad_output111111111111',grad_output[0:2,:,:,:])
        #对梯度为0的部分赋予一个很小的值
        # x=torch.tensor(0.01,device=device) #固定值
        # x=torch.tensor(random.uniform(0.001,0.01),device=device) #随机值
        # grad_output=torch.where(grad_output==0,torch.tensor(random.uniform(0.001,0.01),device=device) ,grad_output) 
       
        '''
        index=torch.where(grad_output==0)
        # data=np.random.uniform(-0.01,0.01,size=(grad_output.shape))
        data=np.random.uniform(-0.01,0.01,size=(grad_output.shape))
        data=torch.from_numpy(data).float().to(device)
        temp=data[index]
        grad_output[index]=temp
        '''
        
        # print('grad_output222222222',grad_output[0:1,:,:,:])
        # print(grad_output[-1:,:,:,:]==0.01) #True
        # grad_input=grad_output

        # 不用可变alpha，记得把最上面的全局alpha注释掉
        # global  alpha
        # alpha+=0.0001
        alpha=my_alpha
        # print('alpha',alpha)
        # exit()
        grad_input=F.softmax(grad_output*alpha,dim=1)
        # grad_input = torch.clamp(grad_output, -1, 1)
        # print('grad_input[0,:,:,:]',grad_input[0,:,:,:])
        # print(grad_input[0,:,:,:]==grad_output[0,:,:,:])
        # print('grad_input[0:2,:,:,:]',grad_input[0:2,:,:,:])
        # t+=1
        # print('t',t)
        '''
        #带温度系数的softmax函数
        t0=10 #初始温度
        T=1 #迭代次数
        tau=t0/(1+math.log(T))
        grad_input=F.softmax(grad_output/tau,dim=0)
        T=T+1
        '''
        #Gumbel_SoftMax
        # grad_input = torch.clamp(grad_output, -1, 1)
        # print('grad_output',grad_output)
        # print(grad_output==0.0)
        # print(grad_output[0,0,0,0]==0.0) #True
        # print('grad_input',grad_input)
        # print('\nx.grad', grad_output)
        return grad_input


sign_incl0 = SignIncludingZero.apply
sign_excl0 = SignExcludingZero.apply
sign_lrgbw = SignLRGBWZero.apply


def less_equal(a, b):  # a <= b
    return sign_incl0(b - a)


def less_than(a, b):  # a < b
    # print('sign_excl0(b - a)',sign_excl0(b - a))
    return sign_excl0(b - a)


def greater_equal(a, b):  # a >= b
    return sign_incl0(a - b)


def greater_than(a, b):  # a > b
    return sign_excl0(a - b)

def produce_rgbw_cfa(x,alpha_0):
    return_alpha(alpha_0)
    return sign_lrgbw(x)
