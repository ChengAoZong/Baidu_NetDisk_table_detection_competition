# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
from tqdm import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression_landmark, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.general import bbox_iou
from utils.torch_utils import select_device, load_classifier, time_synchronized

from scipy.optimize import linear_sum_assignment

from models.yolo import Model
import yaml
from heatmap.get_point import *

# 用于根据规则计算得分 方便自己测试模型的好坏

# precision平均分为两部分，一部分是框，与真值的IOU大于等于0.9记为正确匹配
# 另外一部分是点的预测，预测的点与真实点的距离小于等于20像素记为正确的点
# 最终的精度是二者取平均

# 召回率 IOU大于0.9记为找到
# 点距离小于20为找回
# 最终召回率为二者的平均

# 最终得分为 0.5*precision + 0.5*recall


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords


def show_results(img, xywh, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 3 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def warpimage(img, pts1):
    # right_bottom, left_bottom, left up , left_up
    # pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[400, 200], [0, 200], [0, 0], [400, 0]])
    h, w, c = img.shape
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (400, 200))
    return dst


# 根据图片路径读取下面的每一张图片进行预测，并且获取标签
def getscore(model, dataset_path, device):
    image_list = [os.path.join(dataset_path, 'images', item) for item in os.listdir(dataset_path + 'images\\')]
    label_list = [os.path.join(dataset_path, 'labels', item) for item in os.listdir(dataset_path + 'labels\\')]
    test_length = 100
    image_list = image_list[:test_length]
    label_list = label_list[:test_length]

    landmarks_TP = 0
    landmarks_FP = 0
    total_landmarks = 0
    bbox_TP = 0
    bbox_FP = 0
    total_bbox = 0
    predict_number_is_wrong = 0
    # Load model
    img_size = 640
    conf_thres = 0.3
    iou_thres = 0.5
    time_start = time.time()
    pbar = tqdm(total=len(image_list))
    for k, item in enumerate(image_list):
        pbar.update(test_length/100)
        orgimg = cv2.imread(item)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + dataset_path + 'images\\' + item
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        crop_size = 256
        two_stage_image_size = 512
        pred = model(img)[0] # tensor 1*20160*14
        pred = non_max_suppression_landmark(pred, conf_thres, iou_thres)# list-tensor 1*14
        crop_img = []

        if orgimg.shape[0]>two_stage_image_size and orgimg.shape[1]>two_stage_image_size:
            for i,det in enumerate(pred):# 遍历每一张图片
                print(det)
                print(det.shape)
                det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
                for i in range(det.size()[0]):# 遍历每一张表格
                    px = [det[i][5].item(),det[i][7].item(),det[i][9].item(),det[i][11].item()]
                    py = [det[i][6].item(), det[i][8].item(), det[i][10].item(), det[i][12].item()]
                    start_x = []
                    start_y = []
                    for j in range(4):
                        if px[j] < crop_size/2 or px[j] > (w0-crop_size/2) \
                                or py[j] < crop_size/2 or py[j] > (h0-crop_size/2):
                            if px[j] < crop_size/2:
                                start_lbx = 0
                                end_lbx = crop_size
                                point_x = px[j]
                            elif px[j] > (w0-crop_size/2):
                                end_lbx = w0
                                start_lbx = w0 - crop_size
                                point_x = px[j] - start_lbx
                            else:
                                start_lbx = px[j] - crop_size/2
                                end_lbx = px[j] + crop_size/2
                                point_x = crop_size / 2
                            if py[j] < crop_size/2:
                                start_lby = 0
                                end_lby = crop_size
                                point_y = py[j]
                            elif py[j] > (h0-crop_size/2):
                                end_lby = h0
                                start_lby = h0 - crop_size
                                point_y = py[j]-start_lby
                            else:
                                start_lby = py[j] - crop_size/2
                                end_lby = py[j] + crop_size/2
                                point_y = crop_size / 2
                        else:
                            start_lbx = px[j] - crop_size / 2
                            end_lbx = px[j] + crop_size / 2
                            start_lby = py[j] - crop_size / 2
                            end_lby = py[j] + crop_size / 2
                            point_x = crop_size / 2
                            point_y = crop_size / 2
                        start_x.append(start_lbx)
                        start_y.append(start_lby)
                        crop_img.append(orgimg[int(start_lby):int(end_lby), int(start_lbx):int(end_lbx)])
                #cv2.circle(crop_img[], (int(point_x), int(point_y)), 2, (0,0,255), -1)
                    point_out = get_accuarcy_point(crop_img)    # list4->tuple(w,h)
                    """
                    for i, item in enumerate(crop_img):
                        cv2.circle(crop_img[i], (point_out[i]), 3, (0, 0, 255), -1)
                        cv2.imshow("img", crop_img[i])
                        cv2.waitKey(0)
                    """
                    ori_point = list()
                    for i in range(4):
                        ori_point.append((start_x[i]+point_out[i][0],start_y[i]+point_out[i][1]))
                        cv2.circle(orgimg, (int(ori_point[i][0]), int(ori_point[i][1])), 3, (0, 0, 255), -1)
                    cv2.imshow("ori",orgimg)
                    cv2.waitKey(0)
                    """
                    gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
                    a = np.array([ori_point[0][0],ori_point[0][1],ori_point[1][0],ori_point[1][1],
                                    ori_point[2][0], ori_point[2][1], ori_point[3][0], ori_point[3][1],
                                    ])
                    b = torch.from_numpy(a)
                    c = det[:,5:13]
                    det[:,5:13] = torch.from_numpy(np.array([ori_point[0][0],ori_point[0][1],ori_point[1][0],ori_point[1][1],
                                    ori_point[2][0], ori_point[2][1], ori_point[3][0], ori_point[3][1],
                                    ]))/gn_lks
                    m = torch.tensor((640,640))[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)
                    det[:, 5:13] = det[:,5:13]*torch.tensor(img.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)
                    print(det)
                    print(det.shape)
                    """
        # Process detections
        # 遍历每一张图片
        true_det = []
        true_box_xywh = []
        true_point_lb = []
        true_point_lt = []
        true_point_rt = []
        true_point_rb = []
        # 读取文件获取真值
        with open(label_list[k], 'r') as f:
            for line in f.readlines():
                true_det.append([eval(item) for item in line.split()])
        for i in range(len(true_det)):
            true_box_xywh.append([w0 * true_det[i][1], h0 * true_det[i][2], w0 * true_det[i][3], h0 * true_det[i][4]])
            true_point_lb.append([w0 * true_det[i][5], h0 * true_det[i][6]])
            true_point_lt.append([w0 * true_det[i][7], h0 * true_det[i][8]])
            true_point_rt.append([w0 * true_det[i][9], h0 * true_det[i][10]])
            true_point_rb.append([w0 * true_det[i][11], h0 * true_det[i][12]])

        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            # 预测数量与真实数量相同
            if len(det) == len(true_det):
                total_landmarks = total_landmarks + 4 * len(det)
                total_bbox = total_bbox + 1 * len(det)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

                # 首先使用匈牙利算法进行匹配
                xywh = []
                cost_matrix = np.zeros((len(det), len(det)), dtype=np.float32)
                for i in range(len(det)):
                    for j in range(len(det)):
                        xywh_norm = (xyxy2xywh(torch.as_tensor(det[i, :4]).view(1, 4)) / gn).view(-1).tolist()
                        xywh = [xywh_norm[0] * w0, xywh_norm[1] * h0, xywh_norm[2] * w0, xywh_norm[3] * h0]
                        iou = bbox_iou(torch.tensor(true_box_xywh[j]), torch.tensor(xywh), x1y1x2y2=False, GIoU=False,
                                       DIoU=False, CIoU=False, SIoU=False, eps=1e-7)
                        cost_matrix[i][j] = 1 - iou
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # 其中row_ind表示预测值的顺序 col_ind表示对应的真实值的序号

                # 遍历每一个预测 一个det里面可能包含多个预测框和路标点
                # 按照col_ind取出对应的真实值进行计算
                for i in range(det.size()[0]):
                    xywh_norm = (xyxy2xywh(torch.as_tensor(det[i, :4]).view(1, 4)) / gn).view(-1).tolist()
                    xywh = [xywh_norm[0] * w0, xywh_norm[1] * h0, xywh_norm[2] * w0, xywh_norm[3] * h0]
                    conf = det[i, 4].cpu().numpy()
                    landmarks = (det[i, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                    class_num = det[i, 13].cpu().numpy()

                    # orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
                    # xywh 为归一化的形式 list
                    # box需要的形式为 xywh 当x1y1x2y2为False的时候
                    # bbox返回值为IOU
                    iou = bbox_iou(torch.tensor(true_box_xywh[col_ind[i]]), torch.tensor(xywh), x1y1x2y2=False,
                                   GIoU=False, DIoU=False, CIoU=False, SIoU=False, eps=1e-7)

                    distance_lb_square = (true_point_lb[col_ind[i]][0] - landmarks[0] * w0) * (
                                true_point_lb[col_ind[i]][0] - landmarks[0] * w0) + \
                                         (true_point_lb[col_ind[i]][1] - landmarks[1] * h0) * (
                                                     true_point_lb[col_ind[i]][1] - landmarks[1] * h0)
                    distance_lt_square = (true_point_lt[col_ind[i]][0] - landmarks[2] * w0) * (
                                true_point_lt[col_ind[i]][0] - landmarks[2] * w0) + \
                                         (true_point_lt[col_ind[i]][1] - landmarks[3] * h0) * (
                                                     true_point_lt[col_ind[i]][1] - landmarks[3] * h0)
                    distance_rt_square = (true_point_rt[col_ind[i]][0] - landmarks[4] * w0) * (
                                true_point_rt[col_ind[i]][0] - landmarks[4] * w0) + \
                                         (true_point_rt[col_ind[i]][1] - landmarks[5] * h0) * (
                                                     true_point_rt[col_ind[i]][1] - landmarks[5] * h0)
                    distance_rb_square = (true_point_rb[col_ind[i]][0] - landmarks[6] * w0) * (
                                true_point_rb[col_ind[i]][0] - landmarks[6] * w0) + \
                                         (true_point_rb[col_ind[i]][1] - landmarks[7] * h0) * (
                                                     true_point_rb[col_ind[i]][1] - landmarks[7] * h0)
                    point_distance_square = 400
                    if distance_lb_square <= point_distance_square:
                        landmarks_TP = landmarks_TP + 1
                    else:
                        landmarks_FP = landmarks_FP + 1
                    if distance_lt_square <= point_distance_square:
                        landmarks_TP = landmarks_TP + 1
                    else:
                        landmarks_FP = landmarks_FP + 1
                    if distance_rt_square <= point_distance_square:
                        landmarks_TP = landmarks_TP + 1
                    else:
                        landmarks_FP = landmarks_FP + 1
                    if distance_rb_square <= point_distance_square:
                        landmarks_TP = landmarks_TP + 1
                    else:
                        landmarks_FP = landmarks_FP + 1

                    if iou >= 0.8:
                        bbox_TP = bbox_TP + 1
                    else:
                        bbox_FP = bbox_FP + 1

                    if 0:
                        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
                        thick = 5

                        """
                        cv2.circle(orgimg, (int(true_point_lb[col_ind[i]][0]), int(true_point_lb[col_ind[i]][1])), thick, color[0], -1)
                        cv2.circle(orgimg, (int(true_point_lt[col_ind[i]][0]), int(true_point_lt[col_ind[i]][1])), thick, color[1], -1)
                        cv2.circle(orgimg, (int(true_point_rt[col_ind[i]][0]), int(true_point_rt[col_ind[i]][1])), thick, color[2], -1)
                        cv2.circle(orgimg, (int(true_point_rb[col_ind[i]][0]), int(true_point_rb[col_ind[i]][1])), thick, color[3], -1)
                        """
                        cv2.circle(orgimg, (int(landmarks[0]*w0), int(landmarks[1]*h0)), thick, color[0], -1)
                        cv2.circle(orgimg, (int(landmarks[2]*w0), int(landmarks[3]*h0)), thick, color[1], -1)
                        cv2.circle(orgimg, (int(landmarks[4]*w0), int(landmarks[5]*h0)), thick, color[2], -1)
                        cv2.circle(orgimg, (int(landmarks[6]*w0), int(landmarks[7]*h0)), thick, color[3], -1)

                        cv2.rectangle(orgimg, (
                        int(true_box_xywh[col_ind[i]][0] - 0.5 * true_box_xywh[col_ind[i]][2]), int(true_box_xywh[col_ind[i]][1] - 0.5 * true_box_xywh[col_ind[i]][3])),
                                      (int(true_box_xywh[col_ind[i]][0] + 0.5 * true_box_xywh[col_ind[i]][2]),
                                       int(true_box_xywh[col_ind[i]][1] + 0.5 * true_box_xywh[col_ind[i]][3])), color[4], thickness=1,
                                      lineType=cv2.LINE_AA)

                        cv2.rectangle(orgimg, (
                        int(xywh[0] - 0.5 * xywh[2]), int(xywh[1] - 0.5 * xywh[3])),
                                      (int(xywh[0] + 0.5 * xywh[2]),
                                       int(xywh[1] + 0.5 * xywh[3])), color[1], thickness=1,
                                      lineType=cv2.LINE_AA)


                        #orgimg = cv2.resize(orgimg, (1680, 960))
                        cv2.imshow("table", orgimg)
                        cv2.waitKey(0)

            else:  # 漏检了或者多检了 就当全部检测错误 这样算的得分会稍微低一点
                num = max(len(det), len(true_det))
                landmarks_FP = landmarks_FP + 4 * num
                bbox_FP = bbox_FP + num
                total_landmarks = total_landmarks + 4 * num
                total_bbox = total_bbox + num
                predict_number_is_wrong = predict_number_is_wrong + 1

    time_end = time.time()
    print(f'spent time is {time_end - time_start}')
    precision_landmarks = landmarks_TP / (landmarks_TP + landmarks_FP)
    recall_landmarks = landmarks_TP / total_landmarks
    precision_bbox = bbox_TP / (bbox_TP + bbox_FP)
    recall_bbox = bbox_TP / total_bbox
    final_score = 0.5 * (0.5 * precision_landmarks + 0.5 * precision_bbox) + 0.5 * (
                0.5 * recall_landmarks + 0.5 * recall_bbox)
    print(f'landmarks precision is {landmarks_TP}/{landmarks_TP + landmarks_FP} = {precision_landmarks},'
          f'landmarks recall is {landmarks_TP}/{total_landmarks} = {recall_landmarks},'
          f'bbox precision is {bbox_TP}/{bbox_TP + bbox_FP} = {precision_bbox},'
          f'bbox recall is {bbox_TP}/{total_bbox} = {recall_bbox},final score is {final_score}')
    print(f'predict number is wrong : {predict_number_is_wrong}')
    # Stream results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = 'runs\\train\\exp15\\weights\\last.pt'
    # cfg_path = '.\\models\\yolov5s.yaml'
    model = load_model(weights, device)
    # root = '/home/xialuxi/work/dukto/data/CCPD2020/CCPD2020/images/test/'
    dataset_path = "D:\\competion\\detection\\train\\dataset_divided\\test\\"

    # detect_one(model, image_path, device)
    getscore(model, dataset_path, device)
    print('over')


