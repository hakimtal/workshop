import torch

print(torch.__version__)
import albumentations
import sys

import segmentation_models_pytorch as smp

print(smp.__version__)
# import loss_comb
# %%
# base
import os
import sys
import numpy as np
import time
import csv
import argparse
import math
import pandas as pd
import json

# torch
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
import torch
import torch.nn as nn
# fp16
from torch.cuda.amp import autocast, GradScaler

# third party
import random
import cv2
from tqdm import tqdm
import albumentations
from glob import glob
import unet_zoo

# import segmentation_models_pytorch as smp
# %%
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


set_seed(42)

config_path = sys.argv[1]
configs = json.load(open(config_path, "r"))

# %%
# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """ TBD

    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case', ''))
    day = int(data[1].replace('day', ''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def path2info(row):
    path = row['image_path']
    # print(path)
    path = path.replace("\\", "/")
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case', ''))
    day = int(data[-3].split('_')[1].replace('day', ''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['image_id'] = "case%s_day%s_slice_%s" % (case, day, str(slice_).zfill(4))
    # print(row['image_id'])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


# %%
sub_df = pd.read_csv(configs["sample_submission"])
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(r'../input/test-examle/submission.csv')
    sub_df = sub_df.drop(columns=['class', 'predicted']).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=['class', 'predicted']).drop_duplicates()

sub_df = sub_df.apply(lambda x: get_metadata(x), axis=1)
sub_df.head(5)
# %%
if debug:
    paths = glob(f'../input/test-examle/test/**/*png', recursive=True)
else:
    paths = glob(configs["test_images_path"], recursive=True)

path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.apply(lambda x: path2info(x), axis=1)
path_df.head(5)
# %%
path_df = path_df.merge(sub_df, on=['case', 'day', 'slice'], how='left')
path_df.head(5)
# %%
sys.path.append("../det_models")
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, preprocess_16bit, postprocess, preprocess_video
from torch.utils.data import DataLoader

import os
import json
from tqdm import tqdm

# det model setting
compound_coef = 0
threshold = 0.5
iou_threshold = 0.5

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['att_part']

input_size = 256
det_batch_size = 128

det_model_path = configs["det_model_path"]
det_model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
det_model.load_state_dict(torch.load(det_model_path, map_location='cpu'))
# model.load_state_dict(torch.load(det_model_path))
if use_cuda:
    det_model = det_model.cuda()

det_model.requires_grad_(False)
det_model.eval()

# det part
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

path_df['xmin'] = path_df.slice.map(lambda x: 0)
path_df['xmax'] = path_df.slice.map(lambda x: 0)
path_df['ymin'] = path_df.slice.map(lambda x: 0)
path_df['ymax'] = path_df.slice.map(lambda x: 0)

image_index_list = [i for i in range(path_df.shape[0])]
dataloader = DataLoader(dataset=image_index_list, batch_size=det_batch_size, shuffle=False, num_workers=2)

for idx, items in enumerate(tqdm(dataloader)):
    batch_path = [path_df.iloc[int(item)].image_path for item in items]

    ori_imgs, framed_imgs, framed_metas = preprocess_16bit(batch_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = det_model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    # result
    out_batch = invert_affine(framed_metas, out)

    for idx_o, out_pic in enumerate(out_batch):
        bbox_list = out_pic["rois"]
        src_image_path = batch_path[idx_o]
        if len(bbox_list) == 0:
            print("Not Det BBox", batch_path[idx_o])
            image_metas = framed_metas[idx_o]
            x1, y1, x2, y2 = 0, 0, image_metas[2], image_metas[3]
            path_df.loc[path_df.image_path == src_image_path, 'xmin'] = x1
            path_df.loc[path_df.image_path == src_image_path, 'xmax'] = x2
            path_df.loc[path_df.image_path == src_image_path, 'ymin'] = y1
            path_df.loc[path_df.image_path == src_image_path, 'ymax'] = y2
        else:
            x1, y1, x2, y2 = bbox_list[0].astype(np.int)
            path_df.loc[path_df.image_path == src_image_path, 'xmin'] = x1
            path_df.loc[path_df.image_path == src_image_path, 'xmax'] = x2
            path_df.loc[path_df.image_path == src_image_path, 'ymin'] = y1
            path_df.loc[path_df.image_path == src_image_path, 'ymax'] = y2

del det_model
path_df.head(5)

torch.cuda.empty_cache()
# %%

# %%
aux_params = dict(
    pooling='avg',  # one of 'avg', 'max'
    dropout=0.5,  # dropout ratio, default is None
    activation=None,  # activation function, default is None
    classes=3,  # define number of output labels
)
cls_model_list = []

print("cls model: ", len(cls_model_list))
# %%
cls_sl5_model_list = []

for num_fold in range(5):  # 0.886
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnetv2_l_in21k",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/cls_efv2l_320/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    cls_sl5_model_list.append(model)


print("cls sl5 model: ", len(cls_sl5_model_list))
# %%
for num_fold in range(5):  # 0.884 models
    model = unet_zoo.get_unet_models(unet_type="smp",
                                     encoder_name="efficientnet-b7",
                                     in_ch=5,
                                     out_ch=3
                                     )

    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/cls_efb7_320/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    cls_sl5_model_list.append(model)

print("cls sl5 model: ", len(cls_sl5_model_list))
# %%
for num_fold in range(5):
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnet_b7_ns",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/cls_efb7ns_320/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    cls_sl5_model_list.append(model)

print("cls sl5 model: ", len(cls_sl5_model_list))
# %%
cls_sl5_s352_model_list = []
for num_fold in [0, 1, 2, 3, 4]:
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnetv2_m_in21k",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/cls_efv2m_352/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    cls_sl5_s352_model_list.append(model)

print("cls_sl5_s352_model_list: ", len(cls_sl5_s352_model_list))
# %%
RESIZE_SIZE = 320

test_transform = albumentations.Compose([
    albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

test_transform_slice_5 = albumentations.Compose([
    albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
    albumentations.Normalize(mean=(0.108, 0.108, 0.108, 0.108, 0.108), std=(0.170, 0.170, 0.170, 0.170, 0.170),
                             max_pixel_value=255.0, p=1.0)
])

test_transform_slice_5_s352 = albumentations.Compose([
    albumentations.Resize(352, 352, p=1),
    albumentations.Normalize(mean=(0.108, 0.108, 0.108, 0.108, 0.108), std=(0.170, 0.170, 0.170, 0.170, 0.170),
                             max_pixel_value=255.0, p=1.0)
])


def pooling(data, m, n, key='mean'):
    h, w = data.shape
    img_new = []
    res = []
    for i in range(0, h, m):
        line = []
        for j in range(0, w, n):
            x = data[i:i + m, j:j + n]  # 选取池化区域
            if key == 'mean':  # 平均池化
                res.append(np.sum(x[:, :] / (n * m)))
            elif key == 'max':  # 均值池化
                line.append([np.max(x[:, :, 0]), np.max(x[:, :, 1]), np.max(x[:, :, 2])])
            else:
                return data
        img_new.append(line)

    # img_new = sorted(res)[0:-2]
    # print(img_new)
    return np.array(res, dtype='float32')


def normal_img(img, bbox, avp_k=16):
    img_cp = img.copy()

    imgx = img.copy()
    img_new = imgx

    img_cp = img_cp.astype(np.float32)
    img_cp = (img_cp - np.min(img_new)) / (np.max(img_new) - np.min(img_new)) * 255
    img_cp[img_cp >= 255] = 255
    img_cp = img_cp.astype(np.uint8)

    return img_cp


def pre_precess(image_path):
    image = cv2.imread(image_path, -1)
    image = image.astype('float32')
    mx = np.max(image)
    if mx:
        image /= mx
    image *= 255
    return image.astype(np.uint8)


def pre_precess2(image_path, bbox):
    image = cv2.imread(image_path, -1)
    if image is None:
        return None
    return normal_img(image, bbox)


# %%
def pre_precess(image_path):
    image = cv2.imread(image_path, -1)
    if image is None:
        return None
    image = image.astype('float32')
    mx = np.max(image)
    if mx:
        image /= mx
    image *= 255
    return image.astype(np.uint8)


def pre_precess_show(image_path):
    # print(image_path)
    image = cv2.imread(image_path, -1)
    image = np.tile(image[..., None], [1, 1, 3])
    if image is None:
        return None
    image = image.astype('float32')
    mx = np.max(image)
    if mx:
        image /= mx
    image *= 255

    return image.astype(np.uint8)


def read_stack_3_image(path, slice, bbox):
    baseslice = str(slice).zfill(4)
    # image = cv2.imread(path, -1)
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    pslice = str(int(slice) - 1).zfill(4)
    nslice = str(int(slice) + 1).zfill(4)
    image_path1 = os.path.join(dirname, basename.replace(baseslice, pslice))
    image_path2 = os.path.join(dirname, basename.replace(baseslice, nslice))

    img0 = pre_precess2(image_path1, bbox)
    img1 = pre_precess2(path, bbox)
    img2 = pre_precess2(image_path2, bbox)

    img_stack = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

    if img0 is None:
        img0 = img1.copy()

    if img2 is None:
        img2 = img1.copy()

    img_stack[:, :, 0] = img0
    img_stack[:, :, 1] = img1
    img_stack[:, :, 2] = img2

    return img_stack


def read_stack_5_image(path, slice, bbox):
    baseslice = str(slice).zfill(4)

    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    pslice1 = str(int(slice) - 2).zfill(4)
    pslice2 = str(int(slice) - 1).zfill(4)
    nslice1 = str(int(slice) + 1).zfill(4)
    nslice2 = str(int(slice) + 2).zfill(4)

    image_path0 = os.path.join(dirname, basename.replace(baseslice, pslice1))
    image_path1 = os.path.join(dirname, basename.replace(baseslice, pslice2))
    image_path3 = os.path.join(dirname, basename.replace(baseslice, nslice1))
    image_path4 = os.path.join(dirname, basename.replace(baseslice, nslice2))

    img0 = pre_precess2(image_path0, bbox)
    img1 = pre_precess2(image_path1, bbox)
    img2 = pre_precess2(path, bbox)
    img3 = pre_precess2(image_path3, bbox)
    img4 = pre_precess2(image_path4, bbox)

    img_stack = np.zeros((img2.shape[0], img2.shape[1], 5), dtype=np.uint8)

    if img0 is None:
        img0 = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
    if img1 is None:
        img1 = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
    if img3 is None:
        img3 = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
    if img4 is None:
        img4 = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)

    img_stack[:, :, 0] = img0
    img_stack[:, :, 1] = img1
    img_stack[:, :, 2] = img2
    img_stack[:, :, 3] = img3
    img_stack[:, :, 4] = img4

    return img_stack


class Uwmgi_Dataset_seg_test(data.Dataset):
    def __init__(self,
                 df=None,
                 idx=None,
                 transform=None
                 ):
        self.df = df
        self.idx = np.asarray(idx)
        self.transform = transform
        self.transform_sl5 = test_transform_slice_5
        self.transform_sl5_s352 = test_transform_slice_5_s352

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        index = self.idx[index]
        image_path = self.df.iloc[index].image_path
        image_id = self.df.iloc[index].id

        ymin = self.df.iloc[index].ymin
        ymax = self.df.iloc[index].ymax
        xmin = self.df.iloc[index].xmin
        xmax = self.df.iloc[index].xmax

        bbox = [xmin, ymin, xmax, ymax]
        slice_id = self.df.iloc[index].slice

        image_sl3 = read_stack_3_image(image_path, slice_id, bbox)
        shape = image_sl3.shape

        image_sl5 = read_stack_5_image(image_path, slice_id, bbox)

        image_sl3_crop = image_sl3[ymin:ymax, xmin: xmax, :]
        image_sl5_crop = image_sl5[ymin:ymax, xmin: xmax, :]

        if self.transform is not None:
            augmented_sl3 = self.transform(image=image_sl3_crop)
            image_sl3_ts = augmented_sl3['image'].transpose(2, 0, 1)

        if self.transform_sl5 is not None:
            augmented_sl5 = self.transform_sl5(image=image_sl5_crop)
            image_sl5_ts = augmented_sl5['image'].transpose(2, 0, 1)

        if self.transform_sl5_s352 is not None:
            augmented_sl5_s352 = self.transform_sl5_s352(image=image_sl5_crop)
            image_sl5_s352_ts = augmented_sl5_s352['image'].transpose(2, 0, 1)

        bbox = torch.FloatTensor(bbox)
        shape = torch.FloatTensor(shape)

        return image_sl3_ts, image_sl5_ts, image_sl5_s352_ts, image_id, bbox, shape, image_path


# %%
c_test = np.where((sub_df['id'] != "train"))[0]
test_dataset = Uwmgi_Dataset_seg_test(path_df, c_test, test_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False)


# %%
def post_process_minsize(mask, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(mask.shape, np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# %%
path_df['pos_flag'] = path_df.slice.map(lambda x: 0)
dict_cls_ens = {}
# %%
cls_threshold = 0.5
seg_threshold = 0.4
outputs = []
# (['stomach', 'large_bowel', 'small_bowel']):
# case2_day1_slice_0001,large_bowel,
# case2_day1_slice_0001,small_bowel,
# case2_day1_slice_0001,stomach,
print("debug:", debug)
with torch.no_grad():
    # for index, input_msg in enumerate(test_loader):
    for index, (
    input_batch, input_batch_sl5, input_batch_sl5_s352, id_batch, input_crop, input_shape, input_path) in enumerate(
            tqdm(test_loader)):

        batch_preds = []

        for j in range(15):  # 5fold
            batch_pred, _ = cls_sl5_model_list[j](input_batch_sl5.cuda())
            batch_pred = batch_pred.sigmoid()
            batch_pred = batch_pred.detach().cpu().numpy()
            batch_preds.append(batch_pred)

        for j in range(5):  # 5fold
            batch_pred, _ = cls_sl5_s352_model_list[j](input_batch_sl5_s352.cuda())
            batch_pred = batch_pred.sigmoid()
            batch_pred = F.upsample(batch_pred.detach().cpu().float(), size=(320, 320), mode='bilinear').numpy()
            # batch_pred = batch_pred.detach().cpu().numpy()
            batch_preds.append(batch_pred)

        output_seg = np.mean(batch_preds, 0)

        for idx in range(len(id_batch)):
            id_name = id_batch[idx]
            img_output_seg = output_seg[idx, :, :, :]
            crop_bbox = input_crop[idx]
            image_shape = input_shape[idx, :]
            image_path = input_path[idx]

            xmin, ymin, xmax, ymax = int(crop_bbox[0]), int(crop_bbox[1]), int(crop_bbox[2]), int(crop_bbox[3])

            masks = cv2.threshold(img_output_seg, cls_threshold, 1, cv2.THRESH_BINARY)[1]
            masks = masks.transpose(1, 2, 0).astype(np.uint8)
            masks = cv2.resize(masks, (xmax - xmin, ymax - ymin))

            if np.max(masks) > 0:
                path_df.loc[path_df.image_id == id_name, 'pos_flag'] = 1
            else:
                path_df.loc[path_df.image_id == id_name, 'pos_flag'] = 0

            dict_cls_ens[id_name] = []
            for cidx in range(3):
                if np.max(masks[:, :, cidx]) > 0:
                    dict_cls_ens[id_name].append(1)
                else:
                    dict_cls_ens[id_name].append(0)

            masks_zeros = np.zeros((int(image_shape[0]), int(image_shape[1]), int(image_shape[2])), dtype=np.uint8)

            outputs.append([id_name, "large_bowel", rle_encode(masks_zeros[:, :, 1])])
            outputs.append([id_name, "small_bowel", rle_encode(masks_zeros[:, :, 2])])
            outputs.append([id_name, "stomach", rle_encode(masks_zeros[:, :, 0])])

# %%
del cls_model_list, cls_sl5_model_list, cls_sl5_s352_model_list
# %%
torch.cuda.empty_cache()
# %%
import gc

gc.collect()
# %%
# post process
from copy import deepcopy

dict_data_cp = deepcopy(dict_cls_ens)
sorted_key = sorted(dict_data_cp)
print(sorted_key[0:10])
len_sorted_key = len(sorted_key)
for idx, key in enumerate(sorted_key):
    if idx < 2 or idx > len_sorted_key - 2:
        continue
    for cat in range(3):
        if dict_data_cp[key][cat] == 0:
            if dict_data_cp[sorted_key[idx - 1]][cat] == 1 and dict_data_cp[sorted_key[idx - 2]][cat] == 1 \
                    and dict_data_cp[sorted_key[idx + 1]][cat] == 1 and dict_data_cp[sorted_key[idx + 2]][cat] == 1:
                dict_cls_ens[key][cat] = 1

        if dict_data_cp[key][cat] == 1:
            if dict_data_cp[sorted_key[idx - 1]][cat] == 0 and dict_data_cp[sorted_key[idx - 2]][cat] == 0 \
                    and dict_data_cp[sorted_key[idx + 1]][cat] == 0 and dict_data_cp[sorted_key[idx + 2]][cat] == 0:
                dict_cls_ens[key][cat] = 0

dict_data_cp = deepcopy(dict_cls_ens)
for cat in [0, 2]:
    neg_flag = True
    for idx, key in enumerate(sorted_key):
        if idx > len_sorted_key - 3:
            continue

        if neg_flag:
            if dict_data_cp[sorted_key[idx]][cat] == 1 and dict_data_cp[sorted_key[idx + 1]][cat] == 1 \
                    and dict_data_cp[sorted_key[idx + 2]][cat] == 1:  # 连续三个为1
                neg_flag = False
        else:
            if dict_data_cp[sorted_key[idx]][cat] == 0 and dict_data_cp[sorted_key[idx + 1]][cat] == 0 \
                    and dict_data_cp[sorted_key[idx + 2]][cat] == 0:  # 连续三个为0
                neg_flag = True

        if neg_flag:
            dict_cls_ens[key][cat] = 0
        else:
            dict_cls_ens[key][cat] = 1
# %%
# %%
seg_model_384_list = []
for num_fold in range(5):
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnetv2_l_in21k",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/seg_efv2l_384/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    seg_model_384_list.append(model)


print("384 seg model: ", len(seg_model_384_list))
# %%
# %%
seg_model_416_list = []
for num_fold in range(5):
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnetv2_l_in21k",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/seg_efv2l_v1_416/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    seg_model_416_list.append(model)

#
print("416 seg model: ", len(seg_model_416_list))
# %%
for num_fold in range(5):
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnetv2_l_in21k",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/seg_efv2l_v2_416/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    seg_model_416_list.append(model)

print("416 seg model: ", len(seg_model_416_list))
# %%
for num_fold in range(5):
    model = unet_zoo.get_unet_models(unet_type="timm",
                                     encoder_name="tf_efficientnetv2_m_in21k",
                                     in_ch=5,
                                     out_ch=3
                                     )
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load("./models_snap/seg_efv2m_416/models_swa_%s.pth.tar" % num_fold)
    model.load_state_dict(state['state_dict'])
    model.eval()
    seg_model_416_list.append(model)
#
print("416 seg model: ", len(seg_model_416_list))
# %%
RESIZE_SIZE = 320

test_transform = albumentations.Compose([
    albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

test_transform_384 = albumentations.Compose([
    albumentations.Resize(384, 384, p=1),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

test_transform_sl5_s384 = albumentations.Compose([
    albumentations.Resize(384, 384, p=1),
    albumentations.Normalize(mean=(0.108, 0.108, 0.108, 0.108, 0.108), std=(0.170, 0.170, 0.170, 0.170, 0.170),
                             max_pixel_value=255.0, p=1.0)
])

test_transform_sl5_s416 = albumentations.Compose([
    albumentations.Resize(416, 416, p=1),
    albumentations.Normalize(mean=(0.108, 0.108, 0.108, 0.108, 0.108), std=(0.170, 0.170, 0.170, 0.170, 0.170),
                             max_pixel_value=255.0, p=1.0)
])


# %%
class Uwmgi_Dataset_seg_test(data.Dataset):
    def __init__(self,
                 df=None,
                 idx=None,
                 transform=None
                 ):
        self.df = df
        self.idx = np.asarray(idx)
        self.transform = transform
        self.transform_sl5_s416 = test_transform_sl5_s416
        self.transform_sl5_s384 = test_transform_sl5_s384

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        index = self.idx[index]
        image_path = self.df.iloc[index].image_path
        image_id = self.df.iloc[index].id

        ymin = self.df.iloc[index].ymin
        ymax = self.df.iloc[index].ymax
        xmin = self.df.iloc[index].xmin
        xmax = self.df.iloc[index].xmax

        bbox = [xmin, ymin, xmax, ymax]
        slice_id = self.df.iloc[index].slice

        #         image = read_stack_3_image(image_path, slice_id, bbox)

        image_sl5 = read_stack_5_image(image_path, slice_id, bbox)
        shape = image_sl5.shape

        #         image_src = image[ymin:ymax, xmin: xmax, :]
        image_src_sl5 = image_sl5[ymin:ymax, xmin: xmax, :]

        #         if self.transform is not None:
        #             augmented320 = self.transform(image=image_src)
        #             image320 = augmented320['image'].transpose(2, 0, 1)

        if self.transform_sl5_s416 is not None:
            augmented_sl5_416 = self.transform_sl5_s416(image=image_src_sl5)
            image_sl5_s416 = augmented_sl5_416['image'].transpose(2, 0, 1)

        if self.transform_sl5_s384 is not None:
            augmented_sl5_s384 = self.transform_sl5_s384(image=image_src_sl5)
            image_sl5_s384 = augmented_sl5_s384['image'].transpose(2, 0, 1)

        bbox = torch.FloatTensor(bbox)
        shape = torch.FloatTensor(shape)

        return image_sl5_s416, image_sl5_s384, image_id, bbox, shape, image_path


# %%
c_test = np.where((path_df['pos_flag'] == 1))[0]
test_dataset = Uwmgi_Dataset_seg_test(path_df, c_test, test_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False)

print(len(c_test))
# %%
submit = pd.DataFrame(data=np.array(outputs), columns=["id", "class_", "predicted"])
# %%
# seg_threshold = 0.4
outputs = []

with torch.no_grad():
    # for index, input_msg in enumerate(test_loader):
    for index, (input_batch_sl5_416, input_batch_sl5_s384, id_batch, input_crop, input_shape, input_path) in enumerate(
            tqdm(test_loader)):

        batch_preds_seg = []
        for j in range(14):  # 5fold
            batch_pred_seg, _ = seg_model_416_list[j](input_batch_sl5_416.cuda())
            batch_pred_seg = batch_pred_seg.sigmoid()
            batch_pred_seg = F.upsample(batch_pred_seg.detach().cpu().float(), size=(384, 384), mode='bilinear').numpy()
            # batch_pred_seg = batch_pred_seg.detach().cpu().numpy()
            batch_preds_seg.append(batch_pred_seg)

        for j in range(5):  # 5fold
            batch_pred_seg, _ = seg_model_384_list[j](input_batch_sl5_s384.cuda())
            batch_pred_seg = batch_pred_seg.sigmoid()
            batch_pred_seg = batch_pred_seg.detach().cpu().numpy()
            # batch_pred_seg = F.upsample(batch_pred_seg.detach().cpu().float(), size=(320, 320), mode='bilinear').numpy()
            batch_preds_seg.append(batch_pred_seg)

        output_seg_seg = np.mean(batch_preds_seg, 0)

        for idx in range(len(id_batch)):
            id_name = id_batch[idx]
            crop_bbox = input_crop[idx]
            image_shape = input_shape[idx, :]
            image_path = input_path[idx]

            xmin, ymin, xmax, ymax = int(crop_bbox[0]), int(crop_bbox[1]), int(crop_bbox[2]), int(crop_bbox[3])

            img_output_seg_seg = output_seg_seg[idx, :, :, :]
            masks_seg = cv2.threshold(img_output_seg_seg, seg_threshold, 1, cv2.THRESH_BINARY)[1]
            masks_seg = masks_seg.transpose(1, 2, 0).astype(np.uint8)
            masks_seg = cv2.resize(masks_seg, (xmax - xmin, ymax - ymin))

            masks_zeros = np.zeros((int(image_shape[0]), int(image_shape[1]), 3), dtype=np.uint8)

            for cidx in range(3):
                if dict_cls_ens[id_name][cidx] > 0:
                    save_mask, _ = post_process_minsize(masks_seg[:, :, cidx], 25)
                    masks_zeros[ymin:ymax, xmin: xmax, cidx] = save_mask

            submit.loc[(submit.id == id_name) & (submit.class_ == "large_bowel"), "predicted"] = rle_encode(
                masks_zeros[:, :, 1])
            submit.loc[(submit.id == id_name) & (submit.class_ == "small_bowel"), "predicted"] = rle_encode(
                masks_zeros[:, :, 2])
            submit.loc[(submit.id == id_name) & (submit.class_ == "stomach"), "predicted"] = rle_encode(
                masks_zeros[:, :, 0])
# %%
submit.rename(columns={"class_": "class"}, inplace=True)
submit = pd.DataFrame(submit)
print(submit.head(5))
# %%
# submit = pd.DataFrame(data=np.array(outputs), columns=["id", "class", "predicted"])
# Fix sub error, refers to: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/320541
if not debug:
    sub_df = pd.read_csv(configs["sample_submission"])
    del sub_df['predicted']
    sub_df = sub_df.merge(submit, on=['id', 'class'])
    sub_df.to_csv('submission.csv', index=False)
else:
    sub_df = pd.read_csv('../input/test-examle/submission.csv')
    del sub_df['predicted']
    sub_df = sub_df.merge(submit, on=['id', 'class'])
    sub_df.to_csv('submission.csv', index=False)

sub_df.head(5)
