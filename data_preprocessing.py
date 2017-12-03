import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import cv2

import json
import random

data = pd.read_csv('data/driving_log.csv', dtype={'center': str, 'left': str, 'right': str,
                                                  'steering': np.float32, 'throttle': np.float32,
                                                  'brake': np.float32, 'speed': np.float32},
                                                    skipinitialspace=1)

print(data.dtypes)
X_train = data['center']
y_train = data['steering']
center = data['center']
left = data['left']
right = data['right']
steering = data['steering']
throttle = data['throttle']
speed = data['speed']
brake = data['brake']
print('Training data size = ', len(X_train))
print('Training labels size = ', len(y_train))
print('Training throttle size = ', len(throttle))
print('Training speed size = ', len(speed))

# examine the data
plt.rcParams.update({'font.size': 7})
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_title('Steering')
plt.hist(steering,50);

fig = plt.figure()
ax1 = fig.add_subplot(3,3,1)
ax1.set_title('Throttle')
plt.hist(throttle,50);
ax2 = fig.add_subplot(3,3,2)
ax2.set_title('Brake')
plt.hist(brake,50);
ax3 = fig.add_subplot(3,3,3)
ax3.set_title('Speed')
plt.hist(speed,50);
plt.show()


def rotate_image(image, angle):
    """
    :param image:
    :param angle:
    :return:
    """
    radians = angle * math.pi / 180
    (height, width, channels) = image.shape
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR), radians



def shift_and_crop(filename, steering, amount):
    """
     amount: negative values shift to the left, positive values shift to the right

    :param filename:
    :param steering:
    :param amount:
    :return:
    """
    img = cv2.imread(filename)
    (rows, cols, channels) = img.shape
    M = np.float32([[1, 0, amount], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    crop_img = crop_image(dst)
    steering = steering + amount / 100.0

    return crop_img, steering


def crop_image(dst):
    """
    # original size 320 x 160, final size 300 x 80crop_img
    :param dst:
    :return:
    """
    crop_img = dst[60:140, 10:310]
    return crop_img


print('crop_images function defined')

# test rotate image
img = cv2.imread('./data/'+X_train[0])
cv2.imwrite('img.jpg', img)
# original image
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
plt.imshow(img)

# rotated image
img_rot,img_st = rotate_image(img,0.1*180/math.pi)
cv2.imwrite('img_rot.jpg', img_rot)
ax2 = fig.add_subplot(1,3,2)
plt.imshow(img_rot)
ax2 = fig.add_subplot(1,3,3)

# cropped image
crop_img = crop_image(img_rot)
cv2.imwrite('crop.jpg', crop_img)
plt.imshow(crop_img)
plt.show()

print('Image size:',img.shape)

# test shifting image
img = cv2.imread('./data/'+X_train[0])

filename = './data/'+X_train[0]
steering = y_train[0]
amount = -12

# original image
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
plt.imshow(img)

# shifted and cropped image
img_crop,ster = shift_and_crop (filename,steering,amount)
plt.imshow(img_crop)
plt.show()

print('Original Steering:',steering)
print('Final Steering:',ster)

# Augment data by:
# 1) using only center camera
#     a) crop images to relevant section (remove car and sky)
# 2) shift images to left and right
# 3) duplicate and flip images (invert steering)
#
# From all of above just save those which fall under the desired probability

import random
import sys

random.seed()
count = 0
offset = 0.1  # steering offset for left and right cameras
print('Saving augmented images. Please wait...')
logfile = open('./data/new_data/new_log.csv', 'a')
num_iter = len(center)

for i in range(num_iter):
    # print(i)
    # print(steering[i+1])
    # print(brake[i+1])
    # discard data corresponding to:
    #  - very small steering values;
    #  - braking events;
    #  - very small speed;
    #  - no throttle applied.
    if abs(steering[i]) < 0.0001 or brake[i] > 0 or speed[i] < 5 or throttle[i] < 0.5:
        continue

    new_images = []
    new_steering = []

    # center camera
    filename = 'data/' + center[i]
    img = cv2.imread(filename)

    # crop image to relevant section (remove car and sky)
    crop_img = crop_image(img)
    new_images.append(crop_img)
    new_steering.append(steering[i])

    # create five new images shifted to left
    # and another five shifted to the right
    for k in range(5):
        new_img, new_st = shift_and_crop(filename, steering[i], -15 + k * 3)
        new_images.append(new_img)
        new_steering.append(new_st)

        new_img, new_st = shift_and_crop(filename, steering[i], 15 - k * 3)
        new_images.append(new_img)
        new_steering.append(new_st)

    for j in range(len(new_images)):
        # add flipped images
        new_images.append(cv2.flip(new_images[j], 1))
        new_steering.append((new_steering[j]) * (-1))

    assert (len(new_images) == 22)

    # save according to the probability
    # higher steering have higher probability of being saved
    for j in range(len(new_images)):
        rand_num = random.random()
        # print(rand_num,',',abs(new_steering[j]))
        if (abs(new_steering[j]) + 0.02) >= rand_num:
            # save this image to the final set
            filename = 'IMG/image_' + str(count) + '.jpg'
            count = count + 1
            cv2.imwrite('data/new_data/' + filename, new_images[j])
            logfile.write(filename + ',' + str(new_steering[j]) + '\n')

logfile.close()

print('Total images saved:', count)