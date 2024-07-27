import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):  #创建一个MyDataset类，并继承Dataset
    def __init__(self,path): #初始化
        self.path=path #数据集地址
        self.name=os.listdir(os.path.join(path,'SegmentationClass')) #获取标签所有的名字

    def __len__(self):
        return len(self.name)  #返回文件数，即为数据集个数

    def __getitem__(self, index):
        segment_name=self.name[index]  #xxx.png
        #拼接地址
        segment_path=os.path.join(self.path,'SegmentationClass',segment_name)#从jpg转化成.png
        image_path=os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))
        #开始读图片
        segment_image=keep_image_size_open(segment_path)
        image=keep_image_size_open(image_path)
        return transform(image),transform(segment_image)

if __name__=='__main__':
    data=MyDataset('D:/papers/UNet/code/unet/data/VOC2012')
    print(data[0][0].shape)
    print(data[0][1].shape)