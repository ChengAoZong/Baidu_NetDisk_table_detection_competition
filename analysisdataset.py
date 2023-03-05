# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import os
import cv2

def xyxy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)  # 返回的都是标准化后的值

resolution_between_0_1000 = 0
resolution_between_1000_2000 = 0
resolution_between_2000_3000 = 0
resolution_between_3000_5000 = 0

if __name__ == '__main__':
    #filename = "/home/zca/Downloads/train/annos.txt"
    filename = "D:\\competion\\detection\\train\\annos.txt"
    txt_out_path = "D:\\competion\\detection\\train\\labels"
    with open(filename, 'r') as f:
        json_data = json.load(f)
    #json data 存放的是最外层对像，其内有很多字典
    for item in json_data.keys():# item 表示每张图片
        txt_name = item.split(".")[0] + ".txt"
        res = []
        classes = 0
        img_path = "D:\competion\\detection\\train\\images\\"+item
        img = cv2.imread(img_path,1)
        h,w,chanels = img.shape
        if max(h,w)<1000:
            resolution_between_0_1000 = resolution_between_0_1000 + 1
        elif max(h,w)<2000:
            resolution_between_1000_2000 = resolution_between_1000_2000 + 1
        elif max(h,w)<3000:
            resolution_between_2000_3000 = resolution_between_2000_3000 + 1
        else:
            resolution_between_3000_5000 = resolution_between_3000_5000 + 1
        dw = 1. / w
        dh = 1. / h
        for table in json_data[item]:# table表示每个框
            x1 = table['box'][0]
            y1 = table['box'][1]
            x2 = table['box'][2]
            y2 = table['box'][3]
            lbx = table['lb'][0]*dw
            lby = table['lb'][1]*dh
            ltx = table['lt'][0]*dw
            lty = table['lt'][1]*dh
            rtx = table['rt'][0]*dw
            rty = table['rt'][1]*dh
            rbx = table['rb'][0]*dw
            rby = table['rb'][1]*dh
            xywh = xyxy2xywh((w,h),[x1,y1,x2,y2])
            res.append([classes,xywh,lbx,lby,ltx,lty,rtx,rty,rbx,rby])
    print(f'resolution under 1000 number is : {resolution_between_0_1000}')
    print(f'resolution between 1000-2000 number is : {resolution_between_1000_2000}')
    print(f'resolution between 2000-3000 number is : {resolution_between_2000_3000}')
    print(f'resolution biger than 3000 number is : {resolution_between_3000_5000}')

