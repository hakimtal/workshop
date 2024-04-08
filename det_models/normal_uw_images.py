

import numpy as np
import cv2
import os
from tqdm import tqdm
import base64
import json
import io
import sys


images_path = sys.argv[1]
normal_image_path = sys.argv[2]

image_cnt = -1
case_list = os.listdir(images_path)
print(len(case_list), case_list[0])
for case in tqdm(case_list):
    day_list = os.listdir(images_path + case)
    for day in day_list:
        case_img_path = "%s%s/%s/scans" % (images_path, case, day)
        case_img_list = os.listdir(case_img_path)
        for image_name in case_img_list:
            image_cnt += 1

            fi_path = "%s/%s" % (case_img_path, image_name)
            img = cv2.imread(fi_path, -1)
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

            image_name_data = image_name.split("_")
            image_save_id = "%s_%s_%s" % (day, image_name_data[0], image_name_data[1])

            cv2.imwrite(r"%s/%s.png" % (normal_image_path, image_save_id), img)


#
#
