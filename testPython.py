# coding=utf-8

import os
from math import sin, cos
from PIL import Image
import heapq
import numpy as np
import cv2


def test_np_tile():
    n, k = 30, 5
    x = np.tile(np.arange(0, n), (k, 1))
    print(x)

    y = np.random.rand(10, 5)
    print(y)


def test_hepify():
    data = [1, 5, 3, 2, 9, 5]
    print(data)

    heapq.heapify(data)

    print(data)


def rotate(src, tar, center=(0, 0), theta=0, expand=False):
    """
    nearest neighbor interpolation
    """
    # convert angle in radians
    theta = round(theta*3.14/180, 2)

    img = Image.open(src)
    img_array = np.array(img)

    height, width = img_array.shape
    x, y = center

    # image after rotation
    new_arr = np.zeros((height, width), dtype='uint8')

    if not expand:
        matrix = np.array([[cos(theta), sin(theta), 0],
                           [-sin(theta), cos(theta), 0],
                           [-x * cos(theta) + y * sin(theta) + x,
                            -x * sin(theta) - y * cos(theta) + y, 1]])

        # fill new array in row-major order
        for i in range(height):
            for j in range(width):
                # using homogeneous coordinates
                pos_y, pos_x, _ = np.matmul(np.array([i, j, 1]), matrix)

                if pos_y >= height - 0.5 or pos_y < -0.5 \
                        or pos_x >= width - 0.5 or pos_x < -0.5:
                    new_arr[i, j] = 0
                else:
                    pos_y, pos_x = int(round(pos_y)), int(round(pos_x))
                    new_arr[i, j] = img_array[pos_y, pos_x]
    else:
        print('[Err]: scalling not implemented now.')

    new_img = Image.fromarray(new_arr)
    new_img.save(tar)
    print('=> success.')

    return img_array, new_arr


def test_rotate():
    img_path = './koala.png'

    if not os.path.isfile(img_path):
        print('=> [Err]: invalid file path.')
        return

    tar_path = './koala_rotate.png'

    img, img_rotate = rotate(img_path, tar_path, center=(256, 192), theta=45)

    cv2.imshow('origin', img)
    cv2.imshow('rotate', img_rotate)
    cv2.waitKey()


if __name__ == '__main__':
    # test_hepify()

    test_rotate()

    print('=> test done.')
