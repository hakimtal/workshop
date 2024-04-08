# Core Author: Zylo117
# Script's Author: winter2897 

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

import os
import json
from tqdm import tqdm
import sys
# Video's path
# video_src = 'videotest.mp4'  # set int to use webcam, set str to read from a video file

compound_coef = 0
force_input_size = None  # set None to use default size

threshold = 0.5
iou_threshold = 0.2

use_cuda = False
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [256, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# load model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
# model.load_state_dict(torch.load(r'./logs2/coco/efficientdet-d0_7_1500.pth', map_location='cpu'))
model_path = sys.argv[1]
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model.load_state_dict(torch.load(r'D:\Kaggle\det_v0\logs2\coco\efficientdet-d0_4_1000.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# function for display
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        
        return imgs[i]
# Box
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
dict_mini_bbox = {}
images_all_path = sys.argv[2]
images_all_list = os.listdir(images_all_path)

from torch.utils.data import DataLoader

data = images_all_list # 定义数据集，需要是一个可迭代的对象
# data = [data[i] for i in range(len(data)) if i % 5 == 0]

dataloader = DataLoader(dataset=data, batch_size=4, shuffle=False)

for idx, item in enumerate(tqdm(dataloader)):  # 迭代输出
    # print(idx, item)
    # print(123)
    batch_image_list = item
    batch_path = [images_all_path + image_name for image_name in batch_image_list]

# for idx, image_name in enumerate(tqdm(images_all_list[::-1])):
    # if not idx % 30 == 0:
    #     continue
    # single_image_path = images_all_path + image_name

    # frame preprocessing
    ori_imgs, framed_imgs, framed_metas = preprocess(batch_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # model predict
    # y = torch.cat([x, x], 0)
    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    # result
    out_batch = invert_affine(framed_metas, out)
    # print(out[0]["rois"])
    for idx_o, out_pic in enumerate(out_batch):
        bbox_list = out_pic["rois"]
        # print(bbox_list)
        image_name = batch_image_list[idx_o]
        print(len(bbox_list))
        if len(bbox_list) == 0:

            print("error", image_name)
            continue
        else:
            image_id = image_name.split(".")[0]
            x1, y1, x2, y2 = bbox_list[0].astype(np.int16)
            dict_mini_bbox[image_id] = {}
            dict_mini_bbox[image_id]["ymin"] = int(y1)
            dict_mini_bbox[image_id]["ymax"] = int(y2)
            dict_mini_bbox[image_id]["xmin"] = int(x1)
            dict_mini_bbox[image_id]["xmax"] = int(x2)
        img_show = display([out_pic], [ori_imgs[idx_o]])
        #
        # # show frame by frame
        cv2.imwrite('test/%s' % image_name, img_show)


with open(r"train_add_effdet0.json", 'w') as file:  # 写文件
    file.write(json.dumps(dict_mini_bbox, indent=4))


