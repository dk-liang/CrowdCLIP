import glob
import cv2
import scipy.io as io
import numpy as np
import math
import os
from PIL import Image


def cut_image_test(image, gt, save_path, fname, f, patch_num):
    width, height = image.size
    item_width = int(width / patch_num)
    item_height = int(height / patch_num)
    index = 0
    for i in range(0 ,patch_num)  :
        for j in range(0 ,patch_num):
            img = image.copy()
            kpoint = gt.copy()
            box = ( j *item_width , i *item_height ,( j +1 ) *item_width ,( i +1 ) *item_height)
            gt_crop = kpoint[i *item_height:( i +1 ) *item_height, j *item_width: ( j +1 ) *item_width]
            gt_count = int(np.sum(gt_crop))
            img_crop = img.crop(box)
            img_save = os.path.join(save_path, fname + '_' + str(index) + '.jpg')
            img_crop.save(img_save)
            f.write('{} {}'.format(fname + '_' + str(index) + '.jpg', gt_count))
            f.write('\n')
            index += 1

def cut_image_train(image, gt, save_path, fname, f, patch_num):
    width, height = image.size
    center_x = int(width / 2)
    center_y = int(height / 2)

    item_width = int(width / patch_num)
    item_height = int(height / patch_num)
    index = 0
    for i in range(0 ,6):
        img = image.copy()
        kpoint = gt.copy()
        box = (center_x - (i+1) *item_width , center_y - (i+1) *item_height , center_x + (i+1) *item_width , center_y + (i+1) *item_height)
        gt_crop = kpoint[center_y - (i+1) *item_height:center_y + (i+1) *item_height, center_x - (i+1) *item_width:center_x + (i+1) *item_width]
        gt_count = int(np.sum(gt_crop))
        img_crop = img.crop(box)
        img_save = os.path.join(save_path, fname + '_' + str(index) + '.jpg')
        img_crop.save(img_save)
        f.write('{} {}'.format(fname + '_' + str(index) + '.jpg', gt_count))
        f.write('\n')
        index += 1


img_count = []
img_paths_train = []
img_paths_test = []
img_cuts = []
path_train = "./datasets/ShanghaiTech/part_B_final/train_data/images/"
path_test = "./datasets/ShanghaiTech/part_B_final/test_data/images/"
for img_path in glob.glob(os.path.join(path_train, '*.jpg')):
    img_paths_train.append(img_path)
for img_path in glob.glob(os.path.join(path_test, '*.jpg')):
    img_paths_test.append(img_path)
print(len(img_paths_train), len(img_paths_test))
img_paths_train.sort()
img_paths_test.sort()
# Prepare the inputs
save_path_train = './processed_datasets/SHB/train_data/'
save_path_test = './processed_datasets/SHB/test_data/'
if not os.path.exists(save_path_train):
    os.makedirs(save_path_train)
if not os.path.exists(save_path_train + 'data_list'):
    os.makedirs(save_path_train + 'data_list')
if not os.path.exists(save_path_test):
    os.makedirs(save_path_test)
if not os.path.exists(save_path_test + 'data_list'):
    os.makedirs(save_path_test + 'data_list')
f_count_train = open("./processed_datasets/SHB/train_data/data_list/train.txt", "w+")
f_count_test = open("./processed_datasets/SHB/test_data/data_list/test.txt", "w+")


for img_path in img_paths_train:
    print(img_path)
    img_cuts = []
    fname = os.path.basename(img_path).split('.')[0]
    image = Image.open(img_path)
    Img_data = cv2.imread(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    Gt_data = mat["image_info"][0][0][0][0][0]
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    cut_image_train(image, kpoint, save_path_train, fname, f_count_train, patch_num=12)

for img_path in img_paths_test:
    print(img_path)
    img_cuts = []
    fname = os.path.basename(img_path).split('.')[0]
    image = Image.open(img_path)
    Img_data = cv2.imread(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    Gt_data = mat["image_info"][0][0][0][0][0]
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    cut_image_test(image, kpoint, save_path_test, fname, f_count_test, patch_num=3)
