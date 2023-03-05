from heatmap.models.net import *
import torch
import cv2
from torchvision import transforms
import os
import numpy as np
test_dir = "D:\\competion\\detection\\keypoint256\\test\\images"

tf = transforms.Compose([
        transforms.ToTensor()
])

# 输入一个图像列表，返回该列表中每一张图像上的角点
def get_accuarcy_point(images):
    weights_path = "D:\\learn\\deeplearning\\heatmap\\weights\\epoch99.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet().to(device=device)
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path,map_location=device))
        print(f"load parameters successful !")
    else:
        print(f"failed load parameters !")
    net.eval()

    batch_img = torch.cat((
                    torch.unsqueeze(tf(images[0]), dim=0),
                    torch.unsqueeze(tf(images[1]), dim=0),
                    torch.unsqueeze(tf(images[2]), dim=0),
                    torch.unsqueeze(tf(images[3]), dim=0)
                   ),dim=0)
    output = net(batch_img).squeeze()# output list -> h * w(4*256*256)
    point_output = []
    for item in output:
        h, w = np.where(item == item.max())
        point_output.append((int(w),int(h)))# list4->tuple(w,h)
    return point_output



if __name__ == '__main__':
    img_list = os.listdir(test_dir)
    img_list = img_list[:4]
    images = []
    for item in img_list:
        img_path = os.path.join(test_dir,item)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        images.append(img)
    point_out = get_accuarcy_point(images)
    for i,item in enumerate(img_list):
        img_path = os.path.join(test_dir,item)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        cv2.circle(img, (point_out[i]), 3, (0, 0, 255), -1)
        cv2.imshow("img",img)
        cv2.waitKey(0)
"""
if __name__ == '__main__':
    img_list = os.listdir(test_dir)
    weights_path = "weights\\epoch99.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet().to(device=device)
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path,map_location=device))
        print(f"load parameters successful !")
    else:
        print(f"failed load parameters !")
    net.eval()
    for item in img_list:
        img_path = os.path.join(test_dir,item)
        img = cv2.imread(img_path)
        image = torch.unsqueeze(tf(img), dim=0) # BCHW
        output = net(image).squeeze() #128 128
        h, w = np.where(output==output.max())
        cv2.circle(img, (int(w), int(h)), 3, (0,0,255), -1)
        cv2.imshow("point",img)
        cv2.waitKey(0)
"""