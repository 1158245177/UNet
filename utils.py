from PIL import Image
#等比缩放代码，使图像大小一致
def keep_image_size_open(path,size=(256,256)):
    img=Image.open(path) #读进来
    temp=max(img.size) #取最长边
    mask=Image.new('RGB',(temp,temp),(0,0,0)) #掩码
    mask.paste(img,(0,0)) #粘到左上角
    mask=mask.resize(size)
    return mask