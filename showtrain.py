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
        backupimg = img.copy()
        h,w,chanels = img.shape
        dw = 1. / w
        dh = 1. / h
        thick = 10
        # 颜色为BGR的形式 蓝色左下角 绿色左上角 红色右上角
        color = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
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
            cv2.circle(img, (int(lbx * w), int(lby * h)), thick, color[0], -1)
            cv2.circle(img, (int(ltx * w), int(lty * h)), thick, color[1], -1)
            cv2.circle(img, (int(rtx * w), int(rty * h)), thick, color[2], -1)
            cv2.circle(img, (int(rbx * w), int(rby * h)), thick, color[3], -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color[4], thickness=1, lineType=cv2.LINE_AA)
            img = cv2.resize(img, (1080, 720))
            cv2.imshow("table",img)
            cv2.waitKey(0)

            # 垂直翻转
            backupimg = cv2.flip(backupimg, 0)
            cv2.circle(backupimg, (int(lbx * w), h - int(lby * h)), thick, color[0], -1)
            cv2.circle(backupimg, (int(ltx * w), h - int(lty * h)), thick, color[1], -1)
            cv2.circle(backupimg, (int(rtx * w), h - int(rty * h)), thick, color[2], -1)
            cv2.circle(backupimg, (int(rbx * w), h - int(rby * h)), thick, color[3], -1)
            backupimg = cv2.resize(backupimg, (1080, 720))
            cv2.imshow("table",backupimg)
            cv2.waitKey(0)
