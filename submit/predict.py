import os
import sys
import glob
import json
import cv2
import paddle
import numpy as np
import time
import x2paddle_code

import onnxruntime as ort



def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 2] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[:, 3] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clip(0, img0_shape[1])  # x1
    coords[:, 1].clip(0, img0_shape[0])  # y1
    coords[:, 2].clip(0, img0_shape[1])  # x2
    coords[:, 3].clip(0, img0_shape[0])  # y2
    coords[:, 4].clip(0, img0_shape[1])  # x3
    coords[:, 5].clip(0, img0_shape[0])  # y3
    coords[:, 6].clip(0, img0_shape[1])  # x4
    coords[:, 7].clip(0, img0_shape[0])  # y4
    return coords

def post_process(res):
    # res : [14] 的tensor

    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (paddle.min(box1[:, None, 2:], box2[:, 2:]) - paddle.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



def non_max_suppression_landmark(prediction, conf_thres=0.65, iou_thres=0.01, classes=None, agnostic=False, multi_label=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 13  # number of classes 1
    xc = prediction[..., 4] > conf_thres  # candidates
    #print('xc: ', prediction[..., 4] )

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [paddle.zeros((0, 14))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 13:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]
        box[:, 0:2] = box[:, 0:2] - box[:, 2:4] / 2
        box[:, 2:4] = box[:, 0:2] + box[:, 2:4]

        #landmarks = x[:, 5:15]

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        conf = x[:, 13:].max(1, keepdim=True)
        j = paddle.zeros_like(conf)
        x = paddle.concat((box, conf, x[:, 5:13], j), 1)[conf.reshape((-1, )) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == paddle.to_tensor(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)]
        if x.shape == [14]:
            output[0] = x
            return output
        # Batched NMS
        c = x[:, 13:14] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = paddle.vision.ops.nms(boxes, iou_thres, scores)  # NMS
        i = set(i.numpy())
        i = list(i)
        i = paddle.to_tensor(i)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = paddle.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output

def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    #image_paths = image_paths[:1]

    result = {}

    for image_path in image_paths:
        time_start = time.time()
        filename = os.path.split(image_path)[1]
        # 读图像
        orgimg = cv2.imread(image_path)
        """
        orgimg, _, _ = letterbox(orgimg, auto=False)
        cv2.imshow("ori",orgimg)
        cv2.waitKey(0)
        """

        img0, _, _ = letterbox(orgimg, auto=False)
        img0 = img0.astype(np.float32)
        img0 /= 255
        img = paddle.vision.transforms.to_tensor(img0)
        img = img[None]

        res = x2paddle_code.main(img)
        res = non_max_suppression_landmark(res)[0]

        if len(res) == 0:
            continue
        if res.shape == [14]:
            res = res[None]
        res[:, :4] = scale_coords(img.shape[2:], res[:, :4], orgimg.shape).round()
        res[:, 5:13] = scale_coords_landmarks(img.shape[2:], res[:, 5:13], orgimg.shape).round()
        res = res.numpy()

        crop_size = 256
        two_stage_image_size = 512
        h0,w0 = orgimg.shape[:2]
        crop_img = []
        if orgimg.shape[0]<two_stage_image_size or orgimg.shape[1]<two_stage_image_size:
            for id, _ in enumerate(res):
                # xmin, ymin, xmax, ymax, x1, x2, x3, x4, y1, y2, y3, y4 = post_process(res[id])
                if filename not in result:
                    result[filename] = []
                result[filename].append({
                    "box": [int(res[id][0]), int(res[id][1]), int(res[id][2]), int(res[id][3])],
                    "lb": [int(res[id][5]), int(res[id][6])],
                    "lt": [int(res[id][7]), int(res[id][8])],
                    "rt": [int(res[id][9]), int(res[id][10])],
                    "rb": [int(res[id][11]), int(res[id][12])],
                })
            """
            img_copy = orgimg.copy()
            cv2.rectangle(img_copy, (int(res[id][0]), int(res[id][1])), (int(res[id][2]), int(res[id][3])), (0, 255, 0), 3)
            cv2.circle(img_copy, (int(res[id][5]), int(res[id][6])), 7, (0, 255, 0), -1)
            cv2.circle(img_copy, (int(res[id][7]), int(res[id][8])), 7, (255, 0, 0), -1)
            cv2.circle(img_copy, (int(res[id][9]), int(res[id][10])), 7, (0, 0, 255), -1)
            cv2.circle(img_copy, (int(res[id][11]), int(res[id][12])), 7, (0, 255, 255), -1)
        img_copy = cv2.resize(img_copy, (1280, 720))
            """
        #cv2.imshow('img', img_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        else: # 图片尺寸大于阈值
            for id,_ in enumerate(res):# 遍历每一张表格
            #for i in range(1):  # 遍历每一张表格
                px = [res[id][5].item(), res[id][7].item(), res[id][9].item(), res[id][11].item()]
                py = [res[id][6].item(), res[id][8].item(), res[id][10].item(), res[id][12].item()]
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

                #cv2.imshow(f"{j}", crop_img[j])
                #cv2.waitKey(0)

        #cv2.circle(crop_img[], (int(point_x), int(point_y)), 2, (0,0,255), -1)

            #point_out = get_accuarcy_point(crop_img)    # list4->tuple(w,h)


                batch_img = paddle.concat((paddle.unsqueeze(paddle.vision.transforms.to_tensor(crop_img[0]), axis=0),
                                     paddle.unsqueeze(paddle.vision.transforms.to_tensor(crop_img[1]), axis=0),
                                     paddle.unsqueeze(paddle.vision.transforms.to_tensor(crop_img[2]), axis=0),
                                     paddle.unsqueeze(paddle.vision.transforms.to_tensor(crop_img[3]), axis=0)
                                     ), axis=0)

                #output = heatmap_paddle.heatmap(batch_img).squeeze()  # output list -> h * w(4*256*256)
                #ort_session = ort.InferenceSession("epoch99.onnx")
                #m = batch_img.numpy().astype(np.float32)
                output = ort_session.run(None, {'images': batch_img.numpy().astype(np.float32)})[0]
                crop_img = []
                point_output = []
                for item in output:
                    item = item.squeeze()
                    h, w = np.where(item == item.max())
                    if len(h) > 1 or len(w)> 1:
                        h = h[0]
                        w = w[0]
                    point_output.append((int(w), int(h)))  # list4->tuple(w,h)



                #for i, item in enumerate(crop_img):
                    #cv2.circle(crop_img[i], (point_output[i]), 3, (0, 0, 255), -1)
                    #cv2.imshow(f"img{i}", crop_img[i])
                #cv2.waitKey(0)


                ori_point = list()
                for i in range(4):
                    ori_point.append((start_x[i]+point_output[i][0],start_y[i]+point_output[i][1]))
                    #cv2.circle(orgimg, (int(ori_point[i][0]), int(ori_point[i][1])), 6, (0, 0, 255), -1)
                #orgimg = cv2.resize(orgimg, (1280, 720))
                #cv2.imshow("ori",orgimg)
                #cv2.waitKey(0)
            # xmin, ymin, xmax, ymax, x1, x2, x3, x4, y1, y2, y3, y4 = post_process(res[id])
                if filename not in result:
                    result[filename] = []
                result[filename].append({
                    "box": [int(res[id][0]), int(res[id][1]), int(res[id][2]), int(res[id][3])],
                    "lb": [int(ori_point[0][0]), int(ori_point[0][1])],
                    "lt": [int(ori_point[1][0]), int(ori_point[1][1])],
                    "rt": [int(ori_point[2][0]), int(ori_point[2][1])],
                    "rb": [int(ori_point[3][0]), int(ori_point[3][1])],
                })
    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ort_session = ort.InferenceSession("epoch99.onnx", providers=['CUDAExecutionProvider'])
    process(src_image_dir, save_dir)