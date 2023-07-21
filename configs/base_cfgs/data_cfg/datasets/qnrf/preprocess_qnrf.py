import os
import time

import cv2
import numpy as np
import scipy.io
import scipy.spatial
import scipy.io as io
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
    for i in range(0, 6) :
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

'''please set your dataset path'''
root = './datasets/UCF-QNRF'
img_test_path = root + '/Test/'
img_train_path = root + '/Train/'
save_test_img_path = root + '/test_data/images_2048/'
save_train_img_path = root + '/train_data/images_2048/'

if not os.path.exists(save_train_img_path):
    os.makedirs(save_train_img_path)
if not os.path.exists(save_test_img_path):
    os.makedirs(save_test_img_path)
img_train = []
img_test = []
for file_name in os.listdir(img_train_path):
    if file_name.split('.')[1] == 'jpg':
        img_train.append(file_name)
for file_name in os.listdir(img_test_path):
    if file_name.split('.')[1] == 'jpg':
        img_test.append(file_name)
img_train.sort()
img_test.sort()
print(len(img_train), len(img_test))

save_path_test = './processed_datasets/UCF-QNRF/test_data/'
if not os.path.exists(save_path_test):
    os.makedirs(save_path_test)
if not os.path.exists(save_path_test + 'data_list'):
    os.makedirs(save_path_test + 'data_list')
f_count_test = open("./processed_datasets/UCF-QNRF/test_data/data_list/test.txt", "w+")

save_path_train = './processed_datasets/UCF-QNRF/train_data/'
if not os.path.exists(save_path_train):
    os.makedirs(save_path_train)
if not os.path.exists(save_path_train+'data_list'):
    os.makedirs(save_path_train+'data_list')
f_count_train = open("./processed_datasets/UCF-QNRF/train_data/data_list/train.txt", "w+")


for k in range(len(img_train)):
    Img_data = cv2.imread(img_train_path + img_train[k])
    print(img_train_path + img_train[k])
    rate = 1

    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate = 2048.0 / Img_data.shape[1]
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate = 2048.0 / Img_data.shape[0]

    Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)
    # print(img_train[k], Img_data.shape)
    new_img_path = (save_train_img_path + img_train[k])
    cv2.imwrite(new_img_path, Img_data)

    fname = os.path.basename(img_train_path + img_train[k]).split('.')[0]
    mat_path = os.path.join(img_train_path, fname + '_ann.mat')
    mat = io.loadmat(mat_path)
    Gt_data = mat['annPoints'] * rate
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    image = Image.open(save_train_img_path + img_train[k])
    cut_image_train(image, kpoint, save_path_train, fname, f_count_train, patch_num=12)

for k in range(len(img_test)):
    Img_data = cv2.imread(img_test_path + img_test[k])
    print(img_test_path + img_test[k])
    rate = 1
    # print(img_test[k], Img_data.shape)
    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >=2048:
        rate = 2048.0 / Img_data.shape[1]
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >=2048:
        rate = 2048.0 / Img_data.shape[0]
    # if k%100==0:
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)
    # print(img_test[k], Img_data.shape)
    new_img_path = (save_test_img_path + img_test[k])
    cv2.imwrite(new_img_path, Img_data)

    fname = os.path.basename(img_test_path + img_test[k]).split('.')[0]
    mat_path = os.path.join(img_test_path, fname + '_ann.mat')
    mat = io.loadmat(mat_path)
    Gt_data = mat['annPoints'] * rate
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1
    image = Image.open(save_test_img_path + img_test[k])
    cut_image_test(image, kpoint, save_path_test, fname, f_count_test, patch_num=4)


