# -*- coding: UTF-8 -*-
# @author hjh

import argparse
from pathlib import Path
from utils.swa_utils import swa

import pandas as pd
import numpy as np
from models import zoo
import torch
from torch import nn
import segmentation_models_pytorch as smp
import cv2
import albumentations
import torch.utils.data as data
import torch.nn.functional as F
import json
import sys
import unet_zoo

###
config_path = sys.argv[6]
print(config_path)
configs = json.load(open(config_path, "r"))


RESIZE_SIZE = configs["RESIZE_SIZE"]

train_transform = albumentations.Compose([
        albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
        albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=0.5),
            albumentations.GridDistortion(num_steps=5, distort_limit=0.5),
        ], p=0.5),

        albumentations.OneOf([
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=(3, 7)),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.5),
        albumentations.CoarseDropout(max_height=int(RESIZE_SIZE * 0.2), max_width=int(RESIZE_SIZE * 0.2),
                                     max_holes=1, fill_value=0, mask_fill_value=0, p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45,
                                        border_mode=cv2.BORDER_CONSTANT, p=0.8), #value=(0,0,0)
        albumentations.Normalize(mean=(0.108, 0.108, 0.108, 0.108, 0.108), std=(0.170, 0.170, 0.170, 0.170, 0.170),
                                 max_pixel_value=255.0, p=1.0)
    ])

val_transform = albumentations.Compose([
    albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
    albumentations.Normalize(mean=(0.108, 0.108, 0.108, 0.108, 0.108), std=(0.170, 0.170, 0.170, 0.170, 0.170),
                             max_pixel_value=255.0, p=1.0)
])


class Uwmgi_Dataset_seg_train(data.Dataset):
    def __init__(self,
                 df=None,
                 idx=None,
                 transform=None
                 ):
        self.df = df
        self.idx = np.asarray(idx)
        self.transform = transform
        bbox_dict_path = configs["bbox_dict_path"]
        self.bbox_dict = json.load(open(bbox_dict_path, 'r', encoding="utf-8"))

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, tidx):
        index = self.idx[tidx]
        image_id = self.df.iloc[index].id

        image_path = self.df.iloc[index].image_path

        base_path = image_path.split("/train/")[1]
        image_path = os.path.join(configs["image_path"], base_path)
        mask_path = os.path.join(configs["mask_path"], base_path)

        image = np.load(image_path.replace("png", "npy"))
        # mini bbox process
        ymin = self.bbox_dict[image_id]["ymin"]
        ymax = self.bbox_dict[image_id]["ymax"]
        xmin = self.bbox_dict[image_id]["xmin"]
        xmax = self.bbox_dict[image_id]["xmax"]
        image = image[ymin:ymax, xmin: xmax, :]

        masks = cv2.imread(mask_path, -1)
        masks = masks[ymin:ymax, xmin: xmax]

        masks[masks > 0] = 1

        if self.transform is not None:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image'].transpose(2, 0, 1)
            masks = augmented['mask'].transpose(2, 0, 1)

        cls_label = eval(self.df.iloc[index].cls_mask)
        cls_label = torch.FloatTensor(cls_label)

        return image, masks, cls_label


def load(filename, model):
    #
    print('load {}'.format(filename))

    model = unet_zoo.get_unet_models(unet_type=str(configs["unet_type"]),
                                     encoder_name=str(configs["encoder_name"]),
                                     in_ch=5,
                                     out_ch=3
                                     )

    model = torch.nn.DataParallel(model)

    pretrained_model_path = filename
    state = torch.load(pretrained_model_path)

    data = state['state_dict']
    model.load_state_dict(data)

    return model


import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv_path = configs["csv_path"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help='input directory which contains models')
    parser.add_argument("-o", "--output", type=str, default='swa_model.pth.tar', help='output model file')
    parser.add_argument("-e0", "--epoch0", type=int, default=4, help='choose epoch to swa')
    parser.add_argument("-e1", "--epoch1", type=int, default=9, help='choose epoch to swa')
    parser.add_argument("-e2", "--epoch2", type=int, default=0, help='choose epoch to swa')
    parser.add_argument("-cfg", "--configs", type=str, default="", help='choose model')
    parser.add_argument("-bs", "--batch-size", type=int, default=20, help='batch size')
    parser.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')

    args = parser.parse_args()

    print(args.input)
    print(args.output)
    print(args.configs)
    print(args.epoch0)
    print(args.epoch1)
    print(args.epoch2)
    args.batch_size = int(configs["train_bs"])
    print('bs is', args.batch_size)

    df_all = pd.read_csv(csv_path)

    model = unet_zoo.get_unet_models(unet_type=str(configs["unet_type"]),
                                     encoder_name=str(configs["encoder_name"]),
                                     in_ch=5,
                                     out_ch=3)

    for f_fold in range(5):
        num_fold = f_fold
        print(num_fold)

        df_all = pd.read_csv(csv_path)

        if configs["swa_model"] == "cls":
            c_train = np.where((df_all['fold'] != num_fold) & (df_all['bad_flag'] == 0))[0]
            c_val = np.where((df_all['fold'] == num_fold) & (df_all['bad_flag'] == 0))[0]

        elif configs["swa_model"] == "seg":
            c_train = np.where((df_all['fold'] != num_fold) & (df_all['empty'] == 0) & (df_all['bad_flag'] == 0))[0]
            c_val = np.where((df_all['fold'] == num_fold) & (df_all['empty'] == 0) & (df_all['bad_flag'] == 0))[0]

        else:
            c_train = []
            print("swa_model error in configs")

        train_dataset = Uwmgi_Dataset_seg_train(df_all, c_train, train_transform)
        args.input = args.input.format(num_fold)

        net = swa(load, model, args.input, train_dataset, args.batch_size, args.device,
                  args.epoch0, args.epoch1, args.epoch2, num_fold)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        torch.save({'epoch': 39, 'state_dict': net.state_dict(), 'valLoss': 0,
                    'best_thr_with_no_mask': 0, 'best_dice_with_no_mask': 0,
                    'best_thr_without_no_mask': 0,
                    'best_dice_without_no_mask': 0}, args.output.format(num_fold))  # save path



