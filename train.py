from torch import nn,optim
import  torch
from torch.utils.data import DataLoader
from data import *
from net import *
from  torchvision.utils import save_image



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path='params/unet.pth'
data_path=r'D:\papers\UNet\code\unet\data\VOC2012'
save_path='train_image'

if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=4,shuffle=True)  #看电脑显卡，，如果电脑性能低，会报错，调节batch——size
    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('not successful load weight')

    opt=optim.Adam(net.parameters())
    loss_fun=nn.BCELoss()

    epoch=1
    while True:
        for i,(image,segment_image) in enumerate(data_loader):#遍历数据集
            image,segment_image=image.to(device),segment_image.to(device)

            out_image=net(image)
            train_loss=loss_fun(out_image,segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()
            if i%5==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
            if i%50==0:
                torch.save(net.state_dict(),weight_path)
            _image=image[0]
            _segment_image=segment_image[0]
            _our_image=out_image[0]

            img=torch.stack([_image,_segment_image,_our_image],dim=0)
            save_image(img,f'{save_path}/{i}.png')

        epoch+=1
