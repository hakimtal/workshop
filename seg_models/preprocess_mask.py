
import numpy as np
import pandas as pd
import os
import cv2


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def id2mask(id_, df=None):
    idf = df[df['id']==id_]
    wh = idf[['height','width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    segmentation = eval(idf.segmentation.values[0].replace("nan", "0"))
    # "['large_bowel', 'small_bowel', 'stomach']
    dict_map = {0: 1, 1: 2, 2: 0}  # need check
    for i, class_ in enumerate(range(3)):
        rle = segmentation[i]
        if rle != 0 and not pd.isna(rle):
            mask[..., dict_map[i]] = rle_decode(rle, shape[:2])
    return mask


if __name__ == "__main__":
    df_path = r"./data/train_fold_updata.csv"
    mask_save_path = r"./data/masks/"
    df = pd.read_csv(df_path)
    for i, id_ in enumerate(df[~df.segmentation.isna()].sample(frac=1.0)['id'].unique()):
        mask = id2mask(id_, df=df) * 255
        idf = df[df['id'] == id_]
        case = idf[['case']].iloc[0].case
        day = idf[['day']].iloc[0].day
        # print(case, day, id_)
        mask_name = os.path.basename(idf[['image_path']].iloc[0].image_path)
        mask_save = mask_save_path + "case%s/case%s_day%s/scans/" % (case, case, day)
        if not os.path.exists(mask_save):
            os.makedirs(mask_save)
        cv2.imwrite(mask_save + "%s" % mask_name, mask)

