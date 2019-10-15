# coding:utf-8

import copy
import os
import numpy as np
import cv2
import math


def getClosenessWeight(sigma_g, H, W):
    # 计算空间距离权重模板
    r, c = np.mgrid[0:H:1, 0:W:1]  # 构造三维表
    r -= int((H-1) / 2)
    c -= int((W-1) / 2)
    close_weight = np.exp(-0.5*(np.power(r, 2) +
                                np.power(c, 2))/math.pow(sigma_g, 2))
    return close_weight


def jointBLF(I, H, W, sigma_c, sigma_s, borderType=cv2.BORDER_DEFAULT):

    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight(sigma_c, H, W)

    # 对I进行高斯平滑
    Ig = cv2.GaussianBlur(I, (W, H), sigma_c)

    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)

    # 对原图和高斯平滑的结果扩充边界
    Ip = cv2.copyMakeBorder(I, cH, cH, cW, cW, borderType)
    Igp = cv2.copyMakeBorder(Ig, cH, cH, cW, cW, borderType)

    # 图像矩阵的行数和列数
    rows, cols = I.shape
    i, j = 0, 0

    # 联合双边滤波的结果
    jblf = np.zeros(I.shape, np.float64)
    for r in range(cH, cH+rows, 1):
        for c in range(cW, cW+cols, 1):
            # 当前位置的值
            pixel = Igp[r][c]

            # 当前位置的邻域
            rTop, rBottom = r-cH, r+cH
            cLeft, cRight = c-cW, c+cW

            # 从 Igp 中截取该邻域，用于构建相似性权重模板
            region = Igp[rTop: rBottom+1, cLeft: cRight+1]

            # 通过上述邻域，构建该位置的相似性权重模板
            similarityWeight = np.exp(-0.5*np.power(region -
                                                    pixel, 2.0)) / math.pow(sigma_s, 2.0)

            # 相似性权重模板和空间距离权重模板相乘
            weight = closenessWeight * similarityWeight

            # 将权重归一化
            weight = weight / np.sum(weight)

            # 权重模板和邻域对应位置相乘并求和
            jblf[i][j] = np.sum(Ip[rTop:rBottom+1, cLeft:cRight+1]*weight)

            j += 1
        j = 0
        i += 1
    return jblf


def Test(num_iter=50, eta=0.5, sigma_c=10, sigma_s=10, L=10000, w_size=9):
    """
    @param sigma_c: sigma_color
    @param sigma_s: sigma_space
    implemention of "Spatial-Depth Super Resolution for Range Images"
    """
    # 读取原始图像(经过畸变矫正)和深度图
    src_path = './src.jpg'
    depthMap_path = './depth.jpg'  # Fusion的结果

    joint = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)  # 以原图为联合引导
    depthMap = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)

    # is_resume true
    # depthMap = cv2.imread('./iter_2_10.jpg', cv2.IMREAD_GRAYSCALE)

    if joint is None or depthMap is None:
        print('[Err]: empty input image.')
        return

    DEPTH_H, DEPTH_W = depthMap.shape

    assert joint.shape == depthMap.shape

    # ---------计算sobel, 用二阶sobel代替原图
    # sobel_x = cv2.Sobel(joint, cv2.CV_16S, 1, 0, cv2.BORDER_DEFAULT)
    # sobel_y = cv2.Sobel(joint, cv2.CV_16S, 0, 1, cv2.BORDER_DEFAULT)

    # sobel_xx = cv2.Sobel(sobel_x, cv2.CV_16S, 1, 0, cv2.BORDER_DEFAULT)
    # sobel_yy = cv2.Sobel(sobel_y, cv2.CV_16S, 0, 1, cv2.BORDER_DEFAULT)

    # sobel_xx = cv2.convertScaleAbs(sobel_xx)
    # sobel_yy = cv2.convertScaleAbs(sobel_yy)

    # joint = cv2.addWeighted(sobel_xx, 0.5, sobel_yy, 0.5, 0.0)
    # ---------

    # 都使用float32计算
    joint = joint.astype('float32')
    depthMap = depthMap.astype('float32')

    # -------------------------- 
    # 构造float32的cost volume和cost_cw(new cost volume): H, W, N
    cost_volume = np.zeros((DEPTH_H, DEPTH_W, 256), dtype='float32')
    cost_cw = np.zeros((DEPTH_H, DEPTH_W, 256), dtype='float32')

    THRESH = eta * L
    for iter_i in range(num_iter):
        for d in range(256):  # depth range
            tmp = np.empty((DEPTH_H, DEPTH_W))
            tmp.fill(d)

            cost_tmp = (tmp - depthMap) ** 2
            cost_tmp = np.where(cost_tmp < THRESH, cost_tmp, THRESH)
            cost_volume[:, :, d] = cost_tmp

            # 联合双边滤波
            cost_cw[:, :, d] = cv2.ximgproc.jointBilateralFilter(joint,
                                                                 cost_volume[:, :, d],
                                                                 -1,
                                                                 sigma_c, sigma_s)
            print('Depth hypothesis %d cost filtered' % d)

        # ------------------- 更新depth
        # get min cost along channels(depth hypotheses)
        min_cost = np.min(cost_cw, axis=2)  # f(x): min cost
        min_cost_depths = np.argmin(cost_cw, axis=2)  # x: min cost indices
        # print(min_cost_depths)

        # 亚像素深度估计
        for y in range(DEPTH_H):
            for x in range(DEPTH_W):
                f_d = cost_cw[y][x][min_cost_depths[y][x]]
                f_d_plus = cost_cw[y][x][min_cost_depths[y][x] + 1]
                f_d_minus = cost_cw[y][x][min_cost_depths[y][x] - 1]

                depth = min_cost_depths[y][x] - ((f_d_plus - f_d_minus) / (
                    2.0 * (f_d_plus + f_d_minus - 2.0 * f_d)))
                depthMap[y][x] = depth

        mat2show = copy.deepcopy(depthMap)
        mat2show = np.round(mat2show)
        mat2show = mat2show.astype('uint8')
        cv2.imwrite('./iter_2_%d.jpg' % (iter_i + 1  + 10), mat2show)
        print('=> iter %d done' % (iter_i + 1 + 10))


if __name__ == '__main__':
    Test()
    print('=> Test done.')
