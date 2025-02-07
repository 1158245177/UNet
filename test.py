import os.path

import torch

from net import *
from utils import keep_image_size_open
from  data import *
from torchvision.utils import save_image

net=UNet()#.cuda() #实例化网络

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

_input=input('please input image path:')

img=keep_image_size_open(_input)
img_data=transform(img)#.cuda()
print(img_data.shape)
img_data=torch.unsqueeze(img_data,dim=0)
out=net(img_data)
save_image(out,'result/result.jpg')
print(out)