import os
import random
import math
from tqdm import tqdm
from shutil import copy, move
import time
# 未切分原始目录 包含 images 和 labels 文件夹
raw_dir = "D:\\competion\\detection\\train\\fordivide"

# 划分之后的训练集目录 包含 images 和 labels 文件夹
train_dir = "D:\\competion\\detection\\train\\dataset\\train"


# 划分之后的测试集目录 包含 images 和 labels 文件夹
test_dir = "D:\\competion\\detection\\train\\dataset\\test"

# 训练集所占比例 此处设定为0.95 即有9500张作为训练集
train_ratio = 0.98
train_set = set()

test_ratio = 1 - train_ratio

img_list = os.listdir(os.path.join(raw_dir,'images'))

train_num = round(len(img_list)*train_ratio)
test_num = len(img_list) - train_num

random.seed(0)

print(f"Create test set.")
for i in tqdm(range(test_num)):
    image_idx = random.randint(0,test_num + train_num - i - 1)
    image_name = img_list.pop(image_idx)
    src_image_dir = os.path.join(raw_dir,"images",image_name)
    dest_image_dir = os.path.join(test_dir, "images", image_name)
    src_label_dir = os.path.join(raw_dir,"labels",image_name.split('.')[0]+'.txt')
    dest_label_dir = os.path.join(test_dir,'labels',image_name.split('.')[0]+".txt")
    move(src_image_dir,dest_image_dir)
    move(src_label_dir,dest_label_dir)

print(f"Create train set.")
train_img_list = os.listdir(os.path.join(raw_dir,"images"))
time.sleep(1)

for item in tqdm(train_img_list):
    src_image_dir = os.path.join(raw_dir, "images", item)
    dest_image_dir = os.path.join(train_dir, "images", item)
    src_label_dir = os.path.join(raw_dir, "labels", item.split('.')[0]+'.txt')
    dest_label_dir = os.path.join(train_dir, 'labels', item.split('.')[0]+'.txt')
    move(src_image_dir, dest_image_dir)
    move(src_label_dir, dest_label_dir)

