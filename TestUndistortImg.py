# coding: utf-8

import numpy as np
import cv2
import os


def undistort(img_path,
              fx, fy, cx, cy,
              k1, k2, p1, p2):
    """
    undistort image using distort model
    test gray-scale image only
    """
    if not os.path.isfile(img_path):
        print('=> [Err]: invalid image path')
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    H, W = img.shape
    img_undistort = np.zeros(img.shape)

    # fill in each pixel in un-distorted image
    for v in range(H):
        for u in range(W):
            x1 = (u - cx) / fx
            y1 = (v - cy) / fy

            r_square = (x1 * x1) + (y1 * y1)
            r_quadric = r_square * r_square

            x2 = x1 * (1.0 + k1 * r_square + k2 * r_quadric) + \
                2.0 * p1 * x1 * y1 + p2 * (r_square + 2.0 * x1 * x1)
            y2 = y1 * (1.0 + k1 * r_square + k2 * r_quadric) + \
                p1 * (r_square + 2.0 * y1 * y1) + 2.0 * p2 * x1 * y1

            # nearest neighbor interpolation
            u_distort = int(fx * x2 + cx)
            v_distort = int(fy * y2 + cy)

            if u_distort < 0 or u_distort >= W \
                    or v_distort < 0 or v_distort >= H:
                img_undistort[v, u] = 0
            else:
                img_undistort[v, u] = img[v_distort, u_distort]

    return img_undistort.astype('uint8')


def test_img_undistortion():
    k1 = -0.28340811
    k2 = 0.07395907
    p1 = 0.00019359
    p2 = 1.76187114e-05
    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375

    img_path = './tests/distorted.png'
    img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_undistort = undistort(img_path,
                              fx, fy, cx, cy,
                              k1, k2, p1, p2)

    cv2.imshow('origin', img_orig)
    cv2.imshow('undistort', img_undistort)

    cv2.waitKey()


if __name__ == '__main__':
    test_img_undistortion()
    print('=> Test done.')


# Ref: https://blog.csdn.net/weixin_39752599/article/details/82389555
