
import torch
print(torch.__version__)
import albumentations
import sys
sys.path.append("./utils")
sys.path.append("./models")
import zoo
import loss_comb

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
import copy
import random
import cv2
from tqdm import tqdm
import albumentations
import segmentation_models_pytorch as smp
from timm.utils.model_ema import ModelEmaV2
import unet_zoo
print("smp version: ", smp.__version__)


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

###
config_path = sys.argv[1]
configs = json.load(open(config_path, "r"))

RESIZE_SIZE = configs["RESIZE_SIZE"]

train_transform = albumentations.Compose([
        albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
        albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
        ]),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=0.5),
            albumentations.GridDistortion(num_steps=5, distort_limit=0.5),
            # albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.5),
        ], p=0.5),

        albumentations.HorizontalFlip(p=0.5),

        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45,
                                        border_mode=cv2.BORDER_CONSTANT, p=0.8),

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


class Uwmgi_Dataset_seg_val(data.Dataset):
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

    def __getitem__(self, index):
        index = self.idx[index]
        image_path = self.df.iloc[index].image_path
        image_id = self.df.iloc[index].id

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


def generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size, c_val, val_transform,
                                    val_batch_size, workers):
    train_dataset = Uwmgi_Dataset_seg_train(df_all, c_train, train_transform)
    val_dataset = Uwmgi_Dataset_seg_val(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False)

    return train_loader, val_loader


class WarmRestart(lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for
                base_lr in self.base_lrs]


def warm_restart(scheduler, T_mult=2):
    if scheduler.last_epoch == scheduler.T_max:
        scheduler.last_epoch = -1
        scheduler.T_max *= T_mult
    return scheduler


def calc_dice(gt_seg, pred_seg):
    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_dice_all(preds_m, ys):
    dice_all = 0
    for cat in range(3):
        for i in range(preds_m.shape[0]):
            pred = preds_m[i, cat, :, :]
            gt = ys[i, cat, :, :]

            if np.sum(gt) == 0 and np.sum(pred) == 0:
                dice_all = dice_all + 1
            elif np.sum(gt) == 0 and np.sum(pred) != 0:
                dice_all = dice_all
            else:
                dice_all = dice_all + calc_dice(gt, pred)

    return dice_all / (preds_m.shape[0] * 3)


def calc_dice_pos(preds_m, ys):
    dice_all = 0
    pos_sample = 0
    for cat in range(3):
        for i in range(preds_m.shape[0]):
            pred = preds_m[i, cat, :, :]
            gt = ys[i, cat, :, :]

            if np.sum(gt) == 0 and np.sum(pred) == 0:
                continue
                # dice_all = dice_all + 1
            elif np.sum(gt) == 0 and np.sum(pred) != 0:
                continue
                # dice_all = dice_all
            else:
                dice_all = dice_all + calc_dice(gt, pred)
                pos_sample += 1

    return dice_all / (pos_sample)


def epochVal(model, dataLoader, loss_seg, loss_cls, c_val, val_batch_size, cls_weight, seg_weight):
    model.eval()
    lossVal = 0
    lossValNorm = 0
    valLoss_seg = 0
    valLoss_cls = 0

    outGT = []
    outPRED = []
    outGT_cls = torch.FloatTensor().cuda()
    outPRED_cls = torch.FloatTensor().cuda()
    with torch.no_grad():
        for i, (input, target_seg, target_cls) in enumerate(dataLoader):
            if i == 0:
                ss_time = time.time()
            print(
                str(i) + '/' + str(int(len(c_val) / val_batch_size)) + '     ' + str((time.time() - ss_time) / (i + 1)),
                end='\r')
            target_cls = target_cls.cuda()
            outGT_cls = torch.cat((outGT_cls, target_cls), 0)
            varInput = torch.autograd.Variable(input).cuda()
            varTarget_seg = torch.autograd.Variable(target_seg.contiguous().cuda())
            varTarget_cls = torch.autograd.Variable(target_cls.contiguous().cuda())

            outGT_bt = F.upsample(varTarget_seg.data.cpu().float(), size=(96, 96), mode='bilinear')
            outGT.append(outGT_bt)
            varOutput_seg, varOutput_cls = model(varInput)
            varTarget_seg = varTarget_seg.float()
            lossvalue_seg = loss_seg(varOutput_seg, varTarget_seg, varTarget_cls)

            valLoss_seg = seg_weight * valLoss_seg + lossvalue_seg.item()
            lossvalue_cls = loss_cls(varOutput_cls, varTarget_cls)
            valLoss_cls = cls_weight * valLoss_cls + lossvalue_cls.item()
            varOutput_seg = varOutput_seg.sigmoid()
            varOutput_cls = varOutput_cls.sigmoid()

            outPRED_cls = torch.cat((outPRED_cls, varOutput_cls.data), 0)

            outPRED_bt = F.upsample(varOutput_seg.data.cpu().float(), size=(96, 96), mode='bilinear')
            outPRED.append(outPRED_bt)
            lossValNorm += 1

            # if i > 10:
            #     break

    valLoss_seg = valLoss_seg / lossValNorm
    valLoss_cls = valLoss_cls / lossValNorm
    valLoss = seg_weight * valLoss_seg + cls_weight * valLoss_cls


    predsr_with_no_mask = F.upsample(torch.cat(outPRED), size=(96, 96), mode='bilinear').numpy()
    ysr_with_no_mask = F.upsample(torch.cat(outGT), size=(96, 96), mode='bilinear').numpy()

    print(predsr_with_no_mask.shape, ysr_with_no_mask.shape)
    dicesr_with_no_mask = []
    thrs = np.arange(0.4, 0.6, 0.02)
    best_dicer_with_no_mask = 0
    best_thrr_with_no_mask = 0

    for i in thrs:
        preds_mr_with_no_mask = (predsr_with_no_mask > i)
        a = calc_dice_all(preds_mr_with_no_mask, ysr_with_no_mask)
        dicesr_with_no_mask.append(a)
    dicesr_with_no_mask = np.array(dicesr_with_no_mask)
    best_dicer_with_no_mask = dicesr_with_no_mask.max()
    best_thrr_with_no_mask = thrs[dicesr_with_no_mask.argmax()]

    index = np.sum(np.reshape(ysr_with_no_mask, (ysr_with_no_mask.shape[0], -1)), 1)
    predsr_without_no_mask = predsr_with_no_mask[index != 0, :, :, :]
    ysr_without_no_mask = ysr_with_no_mask[index != 0, :, :, :]

    dicesr_without_no_mask = []
    thrs = np.arange(0.4, 0.6, 0.02)
    best_dicer_without_no_mask = 0
    best_thrr_without_no_mask = 0

    for i in thrs:
        preds_mr_without_no_mask = (predsr_without_no_mask > i)
        a = calc_dice_pos(preds_mr_without_no_mask, ysr_without_no_mask)
        dicesr_without_no_mask.append(a)
    dicesr_without_no_mask = np.array(dicesr_without_no_mask)

    best_dicer_without_no_mask = dicesr_without_no_mask.max()
    best_thrr_without_no_mask = thrs[dicesr_without_no_mask.argmax()]
    del preds_mr_with_no_mask, ysr_with_no_mask, preds_mr_without_no_mask, ysr_without_no_mask

    return valLoss, valLoss_seg, valLoss_cls, best_thrr_with_no_mask, best_dicer_with_no_mask, best_thrr_without_no_mask, best_dicer_without_no_mask


class SoftDiceLoss_binary(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        target = target.clone().view(batch_size, -1)

        inter = torch.sum(input * target, 1) + smooth
        union = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth

        score = torch.sum(2.0 * inter / union) / float(batch_size)
        score = 1.0 - torch.clamp(score, 0.0, 1.0 - 1e-7)

        return score


class MixLoss_binary(nn.Module):
    def __init__(self):
        super(MixLoss_binary, self).__init__()
        self.bce_loss =  torch.nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss_binary()

    def forward(self, input, target):
        bce_value = self.bce_loss(input, target)
        dice_value = self.dice_loss(input, target)
        score = 2.7 * bce_value + 0.9 * dice_value
        return score


def do_random_batch_mixup(model, input, masks, clss, loss_seg, loss_cls):
    batch_size = len(input)

    alpha = 1  # 0.2  #0.2,0.4
    gm = np.random.beta(alpha, alpha)
    gamma = np.array([gm] * batch_size)
    gamma = np.maximum(1 - gamma, gamma)
    gm = np.maximum(1 - gm, gm)

    gamma = torch.from_numpy(gamma).float().cuda()
    perm = torch.randperm(batch_size).cuda()
    perm_input = input[perm]
    perm_mask = masks[perm].float()
    per_clss = clss[perm]

    input = input.cuda()
    perm_input = perm_input.cuda()

    gamma = gamma.view(batch_size, 1, 1, 1).cuda()
    mix_input = gamma * input + (1 - gamma) * perm_input

    varOutput_seg, varOutput_cls = model(mix_input)
    varOutput_seg = varOutput_seg.float()

    pos_clss = per_clss + clss
    lossvalue_seg = gm * loss_seg(varOutput_seg, masks.float(), pos_clss) + (1 - gm) * loss_seg(varOutput_seg,
                                                                                                perm_mask.float(),
                                                                                                pos_clss)
    lossvalue_cls = gm * loss_cls(varOutput_cls, clss) + (1 - gm) * loss_cls(varOutput_cls, per_clss)

    return lossvalue_seg, lossvalue_cls


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def do_random_batch_cutmix(model, input, masks, clss, loss_seg, loss_cls):
    batch_size = len(input)
    channel = input.shape[1]

    alpha = 0.4  # 0.2  #0.2,0.4
    gm = np.random.beta(alpha, alpha)
    gamma = np.array([gm] * batch_size)
    gamma = np.maximum(1 - gamma, gamma)
    gm = np.maximum(1 - gm, gm)

    perm = torch.randperm(batch_size).to(input.device)
    perm_input = input[perm]
    perm_mask = masks[perm]
    per_clss = clss[perm]

    pos_clss = per_clss
    for b in range(batch_size):
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), gamma[b])
        input[b, :, bbx1:bbx2, bby1:bby2] = perm_input[perm[b], :, bbx1:bbx2, bby1:bby2]
        masks[b, :, bbx1:bbx2, bby1:bby2] = perm_mask[perm[b], :, bbx1:bbx2, bby1:bby2]
        for c in range(3):
            pos_clss[b][c] = torch.max(masks[b, c, :, :])

    varOutput_seg, varOutput_cls = model(input)

    lossvalue_seg = loss_seg(varOutput_seg.float(), masks.float(), pos_clss)
    lossvalue_cls = loss_cls(varOutput_cls, pos_clss)
    #     lossvalue_cls = gm * loss_cls(varOutput_cls, clss) + (1 - gm) * loss_cls(varOutput_cls, per_clss)

    return lossvalue_seg, lossvalue_cls


class ComboLoss_Pos(nn.Module):
    def __init__(self):
        super(ComboLoss_Pos, self).__init__()
        self.comb_loss_ = loss_comb.ComboLoss({'bce': 0.5, 'dice': 0.5, 'lovasz': 1}, channel_weights=[1, 1, 1])

    #         self.hd_loss = HausdorffLoss()

    def forward(self, input, target, target_fc):
        pos_idx = (target_fc > 0.5)
        input = input[pos_idx, :]
        target = target[pos_idx, :]

        input_pos = input.unsqueeze(1)
        target_pos = target.unsqueeze(1)
        pos_loss = self.comb_loss_(input_pos, target_pos)  # + self.hd_loss(input_pos, target_pos, is_average=True)

        return pos_loss


def train_one_model(fold_id, model, train_batch_size, val_batch_size, lr_init, total_epoch, use_pretrained, pretrained_path,
                    csv_path, snapshot_path, num_workers):
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss', 'best_thr_with_no_mask',
              'best_dice_with_no_mask', 'best_thr_without_no_mask', 'best_dice_without_no_mask']
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    scaler = GradScaler()
    for f_fold in range(1):
        num_fold = fold_id
        print(num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:

            writer = csv.writer(f)
            writer.writerow([num_fold])

        df_all = pd.read_csv(csv_path)
        c_train = np.where((df_all['fold'] != num_fold) & (df_all['empty'] == 0) & (df_all['bad_flag'] == 0))[0]
        c_val = np.where((df_all['fold'] == num_fold) & (df_all['empty'] == 0) & (df_all['bad_flag'] == 0))[0]

        print('train dataset:', len(c_train), '  val dataset:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset c_val_without_no_mask:', len(c_val),
                             '  val dataset c_val_with_no_mask:', len(c_val)])
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])

        train_loader, val_loader = generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size,
                                                                   c_val, val_transform, val_batch_size, num_workers)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)

        model = torch.nn.DataParallel(model)

        if use_pretrained:
            pretrained_model_path = pretrained_path
            print("Load Pretrained Model from:", pretrained_model_path)
            state = torch.load(pretrained_model_path)
            model.load_state_dict(state['state_dict'])


        loss_cls = torch.nn.BCEWithLogitsLoss()
        loss_seg = ComboLoss_Pos()

        cls_weight, seg_weight = 0.01, 1

        trMaxEpoch = total_epoch
        lossMIN = 100000
        val_dice_max = 0

        for epochID in range(0, trMaxEpoch):

            start_time = time.time()
            model.train()
            trainLoss = 30
            lossTrainNorm = 0
            trainLoss_cls = 0
            trainLoss_seg = 0

            if epochID <= 5:
                optimizer.param_groups[0]['lr'] = lr_init
            elif 5 < epochID <= 39:
                if epochID != 0:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2)
            elif epochID > 39 and epochID < 42:
                optimizer.param_groups[0]['lr'] = 1e-5
            else:
                optimizer.param_groups[0]['lr'] = 5e-6

            for batchID, (input, target_seg, target_cls) in tqdm(enumerate(train_loader)):
                if batchID == 0:
                    ss_time = time.time()
                optimizer.zero_grad()
                with autocast():
                    varInput = torch.autograd.Variable(input).cuda()
                    varTarget_seg = torch.autograd.Variable(target_seg.contiguous().cuda())
                    varTarget_cls = torch.autograd.Variable(target_cls.contiguous().cuda())

                    r_num = np.random.rand()
                    if r_num < 0.333:
                        lossvalue_seg, lossvalue_cls = do_random_batch_cutmix(model, varInput, varTarget_seg,
                                                                              varTarget_cls,
                                                                              loss_seg, loss_cls)
                    elif r_num > 0.666:
                        lossvalue_seg, lossvalue_cls = do_random_batch_mixup(model, varInput, varTarget_seg,
                                                                             varTarget_cls,
                                                                             loss_seg, loss_cls)
                    else:
                        varOutput_seg, varOutput_cls = model(varInput)
                        varTarget_seg = varTarget_seg.float()

                        lossvalue_seg = loss_seg(varOutput_seg, varTarget_seg, varTarget_cls)
                        lossvalue_cls = loss_cls(varOutput_cls, varTarget_cls)

                    trainLoss_seg = seg_weight * trainLoss_seg + lossvalue_seg.item()
                    trainLoss_cls = cls_weight * trainLoss_cls + lossvalue_cls.item()

                    lossvalue = cls_weight * lossvalue_cls + seg_weight * lossvalue_seg
                    lossTrainNorm = lossTrainNorm + 1

                    scaler.scale(lossvalue).backward()
                    # 防止梯度Nan
                    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    # grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()

            trainLoss_seg = trainLoss_seg / lossTrainNorm
            trainLoss_cls = trainLoss_cls / lossTrainNorm
            trainLoss = seg_weight * trainLoss_seg + cls_weight * trainLoss_cls

            valLoss, valLoss_seg, valLoss_cls = 0, 0, 0
            best_thrr_with_no_mask, best_dicer_with_no_mask = 0, 0
            best_thrr_without_no_mask, best_dicer_without_no_mask = 0, 0

            if (epochID + 1) % 5 == 0 or epochID > 39 or epochID == 0:
                valLoss, valLoss_seg, valLoss_cls, best_thrr_with_no_mask, best_dicer_with_no_mask, best_thrr_without_no_mask, best_dicer_without_no_mask = epochVal(
                    model, val_loader, loss_seg, loss_cls, c_val, val_batch_size, cls_weight, seg_weight)
            epoch_time = time.time() - start_time
            if (epochID + 1) % 5 == 0 and epochID > 15:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'valLoss': valLoss},
                           snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth.tar')

            result = [epochID, round(optimizer.state_dict()['param_groups'][0]['lr'], 6), round(epoch_time, 0),
                      round(trainLoss, 4), round(valLoss, 4), round(best_thrr_with_no_mask, 4),
                      round(best_dicer_with_no_mask, 4),
                      round(best_thrr_without_no_mask, 4), round(best_dicer_without_no_mask, 4)]
            print(result)
            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)

        del model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fold_id = int(sys.argv[2])

    csv_path = configs["csv_path"]
    pretrained = eval(configs["pretrained"])
    snapshot_path = str(configs["snapshot_path"])
    pretrained_model_path = str(configs["pretrained_model_path"]) % fold_id
    print("folds:", fold_id)
    num_workers = int(configs["num_workers"])

    model = unet_zoo.get_unet_models(unet_type=str(configs["unet_type"]),
                                     encoder_name=str(configs["encoder_name"]),
                                     in_ch=5,
                                     out_ch=3
                                     )

    train_bs = int(configs["train_bs"])
    total_epoch = int(configs["total_epoch"])
    lr_init = int(configs["lr_init"])

    train_one_model(fold_id, model, train_bs, train_bs * 2, lr_init, total_epoch,
                    pretrained, pretrained_model_path,
                    csv_path, snapshot_path,
                    num_workers)
