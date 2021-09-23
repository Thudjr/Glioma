import os
from shutil import copyfile
import random
import PIL.Image as Image


train_dir = '/stor2/dingjinrui/glioma/dataset/train/'
valid_dir = '/stor2/dingjinrui/glioma/dataset/validation/'

train_vision_dir = '/stor2/dingjinrui/glioma/dataset/train_vision/'
valid_vision_dir = '/stor2/dingjinrui/glioma/dataset/valid_vision/'

train_imgs_one_vision = '/stor2/dingjinrui/glioma/dataset/train_vision/train_vision.png'
valid_imgs_one_vision = '/stor2/dingjinrui/glioma/dataset/valid_vision/valid_vision.png'


IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 11  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 10  # 图片间隔，也就是合并成一张图后，一共有几列


# 定义图像拼接函数
def image_compose(source_dir,target_path):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1,12):
        files = os.listdir(os.path.join(source_dir,str(y)))
        random.seed(123)
        random.shuffle(files)
        for x,file in enumerate(files[:10]):
            source_img = Image.open(os.path.join(source_dir + str(y), file)).convert('RGB')
            from_image = source_img.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, (x * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))

    return to_image.save(target_path)  # 保存新图



if __name__ == '__main__':
    image_compose(valid_vision_dir,valid_imgs_one_vision)



'''
# copyfile
types = os.listdir(train_dir)
for type in types:
    files = os.listdir(train_dir+type)
    random.seed(123)
    random.shuffle(files)
    for file in files[:100]:
        if not os.path.exists(os.path.join(train_vision_dir,type)):
            os.makedirs(os.path.join(train_vision_dir,type))
        copyfile(os.path.join(train_dir+type,file),os.path.join(train_vision_dir+type,file))

types = os.listdir(valid_dir)
for type in types:
    files = os.listdir(valid_dir+type)
    random.seed(123)
    random.shuffle(files)
    for file in files[:100]:
        if not os.path.exists(os.path.join(valid_vision_dir, type)):
            os.makedirs(os.path.join(valid_vision_dir, type))
        copyfile(os.path.join(valid_dir+type,file),os.path.join(valid_vision_dir+type,file))
'''

