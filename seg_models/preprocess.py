# creat mask, creat stack image

import numpy as np
import cv2
import os
from tqdm import tqdm


def pre_precess(image_path, img_test):
    image = cv2.imread(image_path, -1)
    if image is None:
        return np.zeros((img_test.shape[0], img_test.shape[1]), dtype=np.uint8)
    image = image.astype('float32')
    mx = np.max(image)
    if mx:
        image /= mx
    image *= 255
    return image.astype(np.uint8)


def creat_stack_5_slice(images_path, save_path):
    case_list = os.listdir(images_path)
    for case in tqdm(case_list):
        day_list = os.listdir(images_path + case)
        for day in day_list:
            case_img_path = "%s%s/%s/scans" % (images_path, case, day)
            case_img_list = os.listdir(case_img_path)
            case_img_list = sorted(case_img_list)
            for idx, image_name in enumerate(case_img_list):
                if True:
                    if idx == 0:
                        fi_path_0 = ""
                        fi_path_1 = ""
                        fi_path_2 = "%s/%s" % (case_img_path, case_img_list[idx])
                        fi_path_3 = "%s/%s" % (case_img_path, case_img_list[idx + 1])
                        fi_path_4 = "%s/%s" % (case_img_path, case_img_list[idx + 2])
                    elif idx == 1:
                        fi_path_0 = ""
                        fi_path_1 = "%s/%s" % (case_img_path, case_img_list[idx - 1])
                        fi_path_2 = "%s/%s" % (case_img_path, case_img_list[idx])
                        fi_path_3 = "%s/%s" % (case_img_path, case_img_list[idx + 1])
                        fi_path_4 = "%s/%s" % (case_img_path, case_img_list[idx + 2])
                    elif idx == len(case_img_list) - 2:
                        fi_path_0 = "%s/%s" % (case_img_path, case_img_list[idx - 2])
                        fi_path_1 = "%s/%s" % (case_img_path, case_img_list[idx - 1])
                        fi_path_2 = "%s/%s" % (case_img_path, case_img_list[idx])
                        fi_path_3 = "%s/1%s" % (case_img_path, case_img_list[idx + 1])
                        fi_path_4 = ""
                    elif idx == len(case_img_list) - 1:
                        fi_path_0 = "%s/%s" % (case_img_path, case_img_list[idx - 2])
                        fi_path_1 = "%s/%s" % (case_img_path, case_img_list[idx - 1])
                        fi_path_2 = "%s/%s" % (case_img_path, case_img_list[idx])
                        fi_path_3 = ""
                        fi_path_4 = ""
                    else:
                        fi_path_0 = "%s/%s" % (case_img_path, case_img_list[idx - 2])
                        fi_path_1 = "%s/%s" % (case_img_path, case_img_list[idx - 1])
                        fi_path_2 = "%s/%s" % (case_img_path, case_img_list[idx])
                        fi_path_3 = "%s/%s" % (case_img_path, case_img_list[idx + 1])
                        fi_path_4 = "%s/%s" % (case_img_path, case_img_list[idx + 2])

                    img_test = cv2.imread(fi_path_2, -1)

                    img0 = pre_precess(fi_path_0, img_test)
                    img1 = pre_precess(fi_path_1, img_test)
                    img2 = pre_precess(fi_path_2, img_test)
                    img3 = pre_precess(fi_path_3, img_test)
                    img4 = pre_precess(fi_path_4, img_test)

                    img_stack = np.zeros((img1.shape[0], img1.shape[1], 5), dtype=np.uint8)

                    img_stack[:, :, 0] = img0
                    img_stack[:, :, 1] = img1
                    img_stack[:, :, 2] = img2
                    img_stack[:, :, 3] = img3
                    img_stack[:, :, 4] = img4

                    stack_image_save = save_path + "%s/%s/scans/" % (case, day)
                    if not os.path.exists(stack_image_save):
                        os.makedirs(stack_image_save)
                    np.save(stack_image_save + image_name.replace("png", "npy"), img_stack)


if __name__ == "__main__":

    images_path = r"./data/train/"
    save_path_5slice = r"./data/train_stack_5_slice/"
    creat_stack_5_slice(images_path, save_path_5slice)



