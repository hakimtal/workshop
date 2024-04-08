#
"""
refer:
https://www.kaggle.com/code/awsaf49/uwmgi-mask-data

"""
#
"""
refer:
https://www.kaggle.com/code/awsaf49/uwmgi-mask-data

"""
import pandas as pd
import random
import numpy as np
import os
from glob import glob

from tqdm import tqdm
# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


set_seed(42)
n_fold = 5

df = pd.read_csv('./data/train.csv')
df = df.apply(get_metadata, axis=1)
df.head(5)

paths = glob('./data/train/*/*/*/*')
path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.apply(path2info, axis=1)
df = df.merge(path_df, on=['case','day','slice'])
df.head(5)


# df = pd.read_csv('../input/uwmgi-mask-dataset/uw-madison-gi-tract-image-segmentation/train.csv')
df['empty'] = df.segmentation.map(lambda x: int(pd.isna(x)))

df2 = df.groupby(['id'])['class'].agg(list).to_frame().reset_index()
df2 = df2.merge(df.groupby(['id'])['segmentation'].agg(list), on=['id'])

df = df.drop(columns=['segmentation', 'class'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df.head(5)


skf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=1996)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
    df.loc[val_idx, 'fold'] = int(fold)
df.to_csv("./train_fold.csv")
df.head(5)

# id	case	day	slice	image_path	height	width	empty	class	segmentation	fold
df['cls_mask'] = df.case.map(lambda x: int(pd.isna(x)))
df['bad_flag'] = df.case.map(lambda x: int(pd.isna(x)))
# print(123)
dict_map = {0: 1, 1: 2, 2: 0}  # need check
for index in tqdm(range(df.shape[0])):
    segmentation = df.iloc[index].segmentation
    cls_label = [0, 0, 0]
    empty_flag = 1
    for idx, data in enumerate(segmentation):
        if str(data) == "nan":
            cls_label[dict_map[idx]] = 0
        else:
            empty_flag = 0
            cls_label[dict_map[idx]] = 1

    idx = df.iloc[index].id
    image_id = df.iloc[index].id
    bad_flag = 0
    if image_id.startswith("case7_day0") or image_id.startswith("case81_day30") \
            or image_id.startswith("case138_day0") or image_id.startswith("case43_day18") \
            or image_id.startswith("case43_day26"):
        bad_flag = 1

    df.loc[df.id == idx, 'bad_flag'] = bad_flag
    df.loc[df.id == idx, 'empty'] = empty_flag
    df.loc[df.id == idx, 'cls_mask'] = str(cls_label)

df.to_csv("./data/train_fold_updata.csv")
df.head(5)

