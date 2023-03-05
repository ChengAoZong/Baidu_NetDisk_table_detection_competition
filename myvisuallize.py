import bs4
import sys
from pathlib import Path
import os
import cv2
import torch
import numpy as np
import torchvision
import datetime
from time import strftime
# coding=utf-8
import requests
import re, time

from utils.general import (check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,strip_optimizer, xyxy2xywh)



names = ['table']

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = torch.device('cpu')
model = torch.load('runs\\train\\exp15\\weights\\best.pt',map_location=device)
#model = torch.load('/home/zca/Downloads/yolov5/coco/mymodel/yolov5s.pt',map_location=torch.device('cuda:0'))
model = model['model'].to(device).float()

img = cv2.imread("D:\\competion\\detection\\train\\images\\border_0_00BQSK4CJMM5K3UM477Z.jpg")


conf_thres = 0.5
iou_thres = 0.45
classes = 0
agnostic_nms = False
max_det = 10

recordNum = 10
recordFlag = 0

def mydetect(img):
    im0 = img.copy()
    img,__,__ = letterbox(img)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    im = torch.from_numpy(img).to(device).float()
    #im = im.half()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im,augment=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    #out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
    dets = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls) # 检测目标对应的名称
                # xyxy 包含了目标的坐标
                dets.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), names[c], f'{conf:.2f}'])
                print(names[c])
    return dets,im0
    #dets 存放的内容为 左上角x 左上角y 右下角x 右下角y 类别 置信度

def myVisualize(dets,frame):
    for j in dets:
        cv2.rectangle(frame, (j[0], j[1]), (j[2], j[3]), (0, 255, 0), 3)
        #cv2.circle(frame, (j[0], j[1]), 5, (255, 0, 0), -1, shift=0)
        #cv2.circle(frame, (j[2], j[3]), 5, (0, 0, 255), -1, shift=0)
        centerx = int((j[0] + j[2]) / 2)
        centery = int((j[1] + j[3]) / 2)
        #cv2.circle(frame, (centerx, centery), 10, (0, 0, 255), -1, shift=0)
    cv2.imshow('Find Object', frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    dets,im0 = mydetect(img)
    myVisualize(dets,im0)

