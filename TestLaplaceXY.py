# coding:utf-8

import copy
import os
import numpy as np
import cv2
import math


# ------------- PatchMatch optimization of depth map
def random_search2(depthMap,
                   mask,
                   sobel_xx, sobel_yy,
                   x, y,
                   alpha, sigma_1, sigma_2):
    """
    随机搜索
    """
    global Ed_sum, mat

    # current energy
    if mask[y][x]:
        Ed_current = (depthMap[y][x] - mat[y][x]) ** 2
    else:
        d_2_deriv_x = 2.0 * mat[y][x] - mat[y][x - 1] - mat[y][x + 1]
        d_2_deriv_y = 2.0 * mat[y][x] - mat[y - 1][x] - mat[y + 1][x]
        w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))
        Ed_current = (depthMap[y][x] - mat[y][x]) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    # random depth nergy: random search once
    new_depth = np.random.randint(0, 256)

    if mask[y][x]:
        Ed_new = (depthMap[y][x] - new_depth) ** 2
    else:
        d_2_deriv_x = 2.0 * new_depth - mat[y][x - 1] - mat[y][x + 1]
        d_2_deriv_y = 2.0 * new_depth - mat[y - 1][x] - mat[y + 1][x]
        w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))

        Ed_new = (depthMap[y][x] - new_depth) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    if Ed_new < Ed_new:
        mat[y][x] = new_depth
        Ed_sum += Ed_new - Ed_current
        # print('=> updated Ed_sum: ', Ed_sum)


def propagation2(is_even,
                 depthMap,
                 mask,
                 sobel_xx, sobel_yy,
                 x, y,
                 alpha, sigma_1, sigma_2):
    global Ed_sum, mat

    # current energy
    if mask[y][x]:
        Ed_current = (depthMap[y][x] - mat[y][x]) ** 2
    else:
        d_2_deriv_x = 2.0 * mat[y][x] - mat[y][x - 1] - mat[y][x + 1]
        d_2_deriv_y = 2.0 * mat[y][x] - mat[y - 1][x] - mat[y + 1][x]
        w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))

        Ed_current = (depthMap[y][x] - mat[y][x]) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    # check right, down neighbor
    if is_even:
        # right neighbor(x+1, y) energy
        if mask[y][x]:
            Ed_right = (depthMap[y][x + 1] - mat[y][x + 1]) ** 2
        else:
            d_2_deriv_x = 2.0 * mat[y][x + 1] - mat[y][x] - mat[y][x + 2]
            d_2_deriv_y = 2.0 * mat[y][x + 1] - \
                mat[y - 1][x + 1] - mat[y + 1][x + 1]
            w_x = math.exp(-1.0 * abs(sobel_xx[y][x + 1] / sigma_1))
            w_y = math.exp(-1.0 * abs(sobel_yy[y][x + 1] / sigma_2))

            Ed_right = (depthMap[y][x + 1] - mat[y][x + 1]) ** 2 + \
                alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        # down neighbor(x, y+1) energy
        if mask[y][x]:
            Ed_down = (depthMap[y + 1][x] - mat[y + 1][x]) ** 2
        else:
            d_2_deriv_x = 2.0 * mat[y + 1][x] - \
                mat[y + 1][x - 1] - mat[y + 1][x + 1]
            d_2_deriv_y = 2.0 * mat[y + 1][x] - mat[y][x] - mat[y + 2][x]
            w_x = math.exp(-1.0 * abs(sobel_xx[y + 1][x] / sigma_1))
            w_y = math.exp(-1.0 * abs(sobel_yy[y + 1][x] / sigma_2))

            Ed_down = (depthMap[y + 1][x] - mat[y + 1][x]) ** 2 + \
                alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        idx = np.argmin(np.array([Ed_current, Ed_right, Ed_down]))

        if idx == 1:  # replace current depth with right neighbor
            mat[y][x] = mat[y][x + 1]
            Ed_sum += Ed_right - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)
        elif idx == 2:  # replace current depth with down neighbor
            mat[y][x] = mat[y + 1][x]
            Ed_sum += Ed_down - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)

    # check left, up neighbor
    else:
        # left neighbor(x-1, y) energy
        if mask[y][x]:
            Ed_left = (depthMap[y][x - 1] - mat[y][x - 1]) ** 2
        else:
            d_2_deriv_x = 2.0 * mat[y][x - 1] - mat[y][x - 2] - mat[y][x]
            d_2_deriv_y = 2.0 * mat[y][x - 1] - \
                mat[y - 1][x - 1] - mat[y + 1][x - 1]
            w_x = math.exp(-1.0 * abs(sobel_xx[y][x - 1] / sigma_1))
            w_y = math.exp(-1.0 * abs(sobel_yy[y][x - 1] / sigma_2))

            Ed_left = (depthMap[y][x - 1] - mat[y][x - 1]) ** 2 + \
                alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        # up neighbor(x, y-1) energy
        if mask[y][x]:
            Ed_up = (depthMap[y - 1][x] - mat[y - 1][x]) ** 2
        else:
            d_2_deriv_x = 2.0 * mat[y - 1][x] - \
                mat[y - 1][x - 1] - mat[y - 1][x + 1]
            d_2_deriv_y = 2.0 * mat[y - 1][x] - mat[y - 2][x] - mat[y][x]
            w_x = math.exp(-1.0 * abs(sobel_xx[y - 1][x] / sigma_1))
            w_y = math.exp(-1.0 * abs(sobel_yy[y - 1][x] / sigma_2))

            Ed_up = (depthMap[y - 1][x] - mat[y - 1][x]) ** 2 + \
                alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        idx = np.argmin(np.array([Ed_current, Ed_left, Ed_up]))

        if idx == 1:  # replace current energy with left neighbor
            mat[y][x] = mat[y][x - 1]
            Ed_sum += Ed_left - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)
        elif idx == 2:  # replace current energy with up neighbor
            mat[y][x] = mat[y - 1][x]
            Ed_sum += Ed_up - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)


def propagation(is_even,
                depthMap,
                sobel_xx, sobel_yy,
                x, y,
                alpha, sigma_1, sigma_2):
    global Ed_sum, mat

    # current energy
    d_2_deriv_x = 2.0 * mat[y][x] - mat[y][x - 1] - mat[y][x + 1]
    d_2_deriv_y = 2.0 * mat[y][x] - mat[y - 1][x] - mat[y + 1][x]
    w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
    w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))
    Ed_current = (depthMap[y][x] - mat[y][x]) ** 2 + \
        alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    # check right, down neighbor
    if is_even:
        # right neighbor(x+1, y) energy
        d_2_deriv_x = 2.0 * mat[y][x + 1] - mat[y][x] - mat[y][x + 2]
        d_2_deriv_y = 2.0 * mat[y][x + 1] - \
            mat[y - 1][x + 1] - mat[y + 1][x + 1]
        w_x = math.exp(-1.0 * abs(sobel_xx[y][x + 1] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y][x + 1] / sigma_2))
        Ed_right = (depthMap[y][x + 1] - mat[y][x + 1]) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        # down neighbor(x, y+1) energy
        d_2_deriv_x = 2.0 * mat[y + 1][x] - \
            mat[y + 1][x - 1] - mat[y + 1][x + 1]
        d_2_deriv_y = 2.0 * mat[y + 1][x] - mat[y][x] - mat[y + 2][x]
        w_x = math.exp(-1.0 * abs(sobel_xx[y + 1][x] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y + 1][x] / sigma_2))
        Ed_down = (depthMap[y + 1][x] - mat[y + 1][x]) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        idx = np.argmin(np.array([Ed_current, Ed_right, Ed_down]))

        if idx == 1:  # replace current depth with right neighbor
            mat[y][x] = mat[y][x + 1]
            Ed_sum += Ed_right - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)
        elif idx == 2:  # replace current depth with down neighbor
            mat[y][x] = mat[y + 1][x]
            Ed_sum += Ed_down - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)

    # check left, up neighbor
    else:
        # left neighbor(x-1, y) energy
        d_2_deriv_x = 2.0 * mat[y][x - 1] - mat[y][x - 2] - mat[y][x]
        d_2_deriv_y = 2.0 * mat[y][x - 1] - \
            mat[y - 1][x - 1] - mat[y + 1][x - 1]
        w_x = math.exp(-1.0 * abs(sobel_xx[y][x - 1] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y][x - 1] / sigma_2))
        Ed_left = (depthMap[y][x - 1] - mat[y][x - 1]) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        # up neighbor(x, y-1) energy
        d_2_deriv_x = 2.0 * mat[y - 1][x] - \
            mat[y - 1][x - 1] - mat[y - 1][x + 1]
        d_2_deriv_y = 2.0 * mat[y - 1][x] - mat[y - 2][x] - mat[y][x]
        w_x = math.exp(-1.0 * abs(sobel_xx[y - 1][x] / sigma_1))
        w_y = math.exp(-1.0 * abs(sobel_yy[y - 1][x] / sigma_2))
        Ed_up = (depthMap[y - 1][x] - mat[y - 1][x]) ** 2 + \
            alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

        idx = np.argmin(np.array([Ed_current, Ed_left, Ed_up]))

        if idx == 1:  # replace current energy with left neighbor
            mat[y][x] = mat[y][x - 1]
            Ed_sum += Ed_left - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)
        elif idx == 2:  # replace current energy with up neighbor
            mat[y][x] = mat[y - 1][x]
            Ed_sum += Ed_up - Ed_current
            # print('=> updated Ed_sum: ', Ed_sum)


def random_search(depthMap,
                  sobel_xx, sobel_yy,
                  x, y,
                  alpha, sigma_1, sigma_2):
    """
    随机搜索
    """
    global Ed_sum, mat

    # current energy
    d_2_deriv_x = 2.0 * mat[y][x] - mat[y][x - 1] - mat[y][x + 1]
    d_2_deriv_y = 2.0 * mat[y][x] - mat[y - 1][x] - mat[y + 1][x]
    w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
    w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))
    Ed_current = (depthMap[y][x] - mat[y][x]) ** 2 + \
        alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    # random depth nergy: random search once
    new_depth = np.random.randint(0, 256)

    d_2_deriv_x = 2.0 * new_depth - mat[y][x - 1] - mat[y][x + 1]
    d_2_deriv_y = 2.0 * new_depth - mat[y - 1][x] - mat[y + 1][x]
    w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
    w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))
    Ed_new = (depthMap[y][x] - new_depth) ** 2 + \
        alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    if Ed_new < Ed_new:
        mat[y][x] = new_depth
        Ed_sum += Ed_new - Ed_current
        # print('=> updated Ed_sum: ', Ed_sum)


def Test2(alpha=10.0, sigma_1=0.1, sigma_2=0.1, num_iter=20):
    print('alpha: %.3f, sigma_1: %.3f, sigma_2: %.3f, num_iter: %d'
          % (alpha, sigma_1, sigma_2, num_iter))

    # 读取原始图像(经过畸变矫正)和深度图
    src_path = './src.jpg'
    depthMap_path = './depth.jpg'  # Fusion的结果

    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    depthMap = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)
    if src is None or depthMap is None:
        print('[Err]: empty input image.')
        return

    DEPTH_H, DEPTH_W = depthMap.shape

    assert src.shape == depthMap.shape

    # 随机初始化一张待估计深度图
    global mat
    mat = np.random.randint(low=0, high=256, size=(DEPTH_H, DEPTH_W))
    # print(mat)

    # 通过原图计算x,y二阶导数
    sobel_x = cv2.Sobel(src, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(src, -1, 0, 1, cv2.BORDER_DEFAULT)

    sobel_xx = cv2.Sobel(sobel_x, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_yy = cv2.Sobel(sobel_y, -1, 0, 1, cv2.BORDER_DEFAULT)

    # cv2.imshow('sobel_xx', sobel_xx)
    # cv2.imshow('sobel_yy', sobel_yy)
    # cv2.waitKey()

    # 计算输入深度图的Mask
    mask = depthMap != 0

    # 计算初始能量函数
    global Ed_sum
    Ed_sum = 0.0

    for y in range(DEPTH_H):
        for x in range(DEPTH_W):

            # 数据项
            if mask[y][x]:
                Ed_sum += (depthMap[y][x] - mat[y][x]) ** 2

            else:
                # 平滑项
                if x - 1 < 0 or x + 1 >= DEPTH_W \
                        or y - 1 < 0 or y + 1 >= DEPTH_H:
                    continue
                else:
                    d_2_deriv_x = 2.0 * mat[y][x] - \
                        mat[y][x - 1] - mat[y][x + 1]
                    d_2_deriv_y = 2.0 * mat[y][x] - \
                        mat[y - 1][x] - mat[y + 1][x]
                    w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
                    w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))
                    Ed_sum += alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    print('=> init Ed: ', Ed_sum)

    # -------------------- optimization...
    for iter_i in range(1, num_iter + 1):

        is_even = iter_i % 2 == 0

        for y in range(2, DEPTH_H - 2):
            for x in range(2, DEPTH_W - 2):
                propagation2(is_even,
                             depthMap,
                             mask,
                             sobel_xx, sobel_yy,
                             x, y,
                             alpha, sigma_1, sigma_2)
                random_search2(depthMap,
                               mask,
                               sobel_xx, sobel_yy,
                               x, y,
                               alpha, sigma_1, sigma_2)

        # 随着迭代的进行, 动态调整alpha
        alpha += 5

        # 输出此轮迭代能量函数值
        print('=> Iter %d | Ed_sum: %.3f' % ( iter_i, Ed_sum))

        mat2show = copy.deepcopy(mat)
        mat2show = mat2show.astype('uint8')
        cv2.imwrite('./iter_%d.jpg' % (iter_i), mat2show)

    print('=> Final Ed_sum: ', Ed_sum)
    mat2show = copy.deepcopy(mat)
    mat2show = mat2show.astype('uint8')
    cv2.imshow('mat', mat2show)
    cv2.waitKey()


def Test(alpha=10.0, sigma_1=0.1, sigma_2=0.1, num_iter=20):
    print('alpha: %.3f, sigma_1: %.3f, sigma_2: %.3f, num_iter: %d'
          % (alpha, sigma_1, sigma_2, num_iter))

    # 读取原始图像(经过畸变矫正)和深度图
    src_path = './src.jpg'
    depthMap_path = './depth.jpg'  # Fusion的结果

    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    depthMap = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)
    if src is None or depthMap is None:
        print('[Err]: empty input image.')
        return

    DEPTH_H, DEPTH_W = depthMap.shape

    assert src.shape == depthMap.shape

    # cv2.imshow('src', src)
    # cv2.waitKey()

    # 随机初始化一张待估计深度图
    global mat
    mat = np.random.randint(low=0, high=256, size=(DEPTH_H, DEPTH_W))
    # print(mat)

    # mat = mat.astype('uint8')
    # cv2.imshow('init depth map', mat)
    # cv2.waitKey()

    # 通过原图计算x,y二阶导数
    sobel_x = cv2.Sobel(src, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(src, -1, 0, 1, cv2.BORDER_DEFAULT)

    sobel_xx = cv2.Sobel(sobel_x, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_yy = cv2.Sobel(sobel_y, -1, 0, 1, cv2.BORDER_DEFAULT)

    # cv2.imshow('sobel_xx', sobel_xx)
    # cv2.imshow('sobel_yy', sobel_yy)
    # cv2.waitKey()

    # 计算输入深度图的Mask
    mask = depthMap != 0

    # 计算初始能量函数
    global Ed_sum
    Ed_sum = 0.0

    for y in range(DEPTH_H):
        for x in range(DEPTH_W):

            # 数据项
            Ed_sum += (depthMap[y][x] - mat[y][x]) ** 2

            # 平滑项
            if x - 1 < 0 or x + 1 >= DEPTH_W \
                    or y - 1 < 0 or y + 1 >= DEPTH_H:
                continue
            else:
                d_2_deriv_x = 2.0 * mat[y][x] - mat[y][x - 1] - mat[y][x + 1]
                d_2_deriv_y = 2.0 * mat[y][x] - mat[y - 1][x] - mat[y + 1][x]
                w_x = math.exp(-1.0 * abs(sobel_xx[y][x] / sigma_1))
                w_y = math.exp(-1.0 * abs(sobel_yy[y][x] / sigma_2))
                Ed_sum += alpha * (w_x * d_2_deriv_x + w_y * d_2_deriv_y)

    print('=> init Ed: ', Ed_sum)

    # -------------------- optimization...
    for iter_i in range(1, num_iter + 1):

        is_even = iter_i % 2 == 0

        for y in range(2, DEPTH_H - 2):
            for x in range(2, DEPTH_W - 2):
                propagation(is_even,
                            depthMap,
                            sobel_xx, sobel_yy,
                            x, y,
                            alpha, sigma_1, sigma_2)
                random_search(depthMap,
                              sobel_xx, sobel_yy,
                              x, y,
                              alpha, sigma_1, sigma_2)
        print('=> Iter %d | Ed_sum: %.3f' % (iter_i, Ed_sum))

        mat2show = copy.deepcopy(mat)
        mat2show = mat2show.astype('uint8')
        cv2.imwrite('./iter_%d.jpg' % (iter_i), mat2show)

    print('=> Final Ed_sum: ', Ed_sum)
    mat2show = copy.deepcopy(mat)
    mat2show = mat2show.astype('uint8')
    cv2.imshow('mat', mat2show)
    cv2.waitKey()


def TestPatchMatchOptimize():
    src_path = 'C:/MyColMap/colmap-dev/workspace/SrcMVS/dense_output/images/IMG_2350.JPG'
    depthMap_path = 'C:/MyColMap/colmap-dev/workspace/result/depth_maps/IMG_2350.JPG.geometric.bin.jpg'  # Fusion的结果

    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    depthMap = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)
    if src is None or depthMap is None:
        print('[Err]: empty input image.')
        return

    depth_h, depth_w = depthMap.shape

    src_resize = cv2.resize(src,
                            (depth_w, depth_h),
                            cv2.INTER_CUBIC)

    assert src_resize.shape == depthMap.shape

    # 计算输入深度图的Mask
    mask = depthMap != 0

    # # --------- visualize mask
    # mask = mask.astype('uint8')  # mask由bool转化为int: 0, 1
    # mask[mask == 1] = 255
    # mask_ = cv2.resize(mask,
    #                    (int(depth_w * 0.3), int(depth_h * 0.3)),
    #                    cv2.INTER_CUBIC)
    # cv2.imshow('mask', mask_)
    # cv2.waitKey()

    # 为mask中为False(深度值为0:不存在深度值)的地方
    # 赋随机初始深度值(在真实的depthmap并不完全随机: 在深度值范围内取值更快收敛)
    depthMap_ = copy.deepcopy(depthMap)  # 保持原始深度图不变

    depthMap_[mask == False] = np.random.random_integers(low=0, high=255)

    depthMap_rand = cv2.resize(depthMap_,
                               (int(depth_w * 0.3), int(depth_h * 0.3)),
                               cv2.INTER_CUBIC)
    depthMap_orig = cv2.resize(depthMap,
                               (int(depth_w * 0.3), int(depth_h * 0.3)),
                               cv2.INTER_CUBIC)

    cv2.imshow('origin depth map', depthMap_orig)
    cv2.imshow('rand fill depth map', depthMap_rand)
    cv2.waitKey()


def testContour():
    """
    如何筛选主要轮廓，忽略尺寸较小的轮廓
    """
    img = cv2.imread('c:/src.jpg', cv2.IMREAD_COLOR)
    if img is None:
        print('=> [Err]: empty img.')
        return
    H, W, N = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(img_gray, (7, 7), 0)

    ret, binary = cv2.threshold(blured,
                                70, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    binary_inv = cv2.bitwise_not(binary)

    binary_ = cv2.resize(binary,
                         (int(W * 0.3 + 0.5), int(H * 0.3 + 0.5)),
                         cv2.INTER_CUBIC)
    binary_inv_ = cv2.resize(binary_inv,
                             (int(W * 0.3 + 0.5), int(H * 0.3 + 0.5)),
                             cv2.INTER_CUBIC)
    cv2.imshow('binary', binary_)
    cv2.imshow('binary_inv', binary_inv_)

    img_2, contours, hierarchy = cv2.findContours(binary_inv,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # print(hierarchy)

    # img_2_ = cv2.resize(img_2,
    #                     (int(W * 0.3 + 0.5), int(H * 0.3 + 0.5)),
    #                     cv2.INTER_CUBIC)
    img_ = cv2.resize(img,
                      (int(W * 0.3 + 0.5), int(H * 0.3 + 0.5)),
                      cv2.INTER_CUBIC)

    cv2.imshow("img_", img_)
    cv2.waitKey()


def testSobel():
    img_path = 'c:/src.jpg'

    if not os.path.isfile(img_path):
        print('[Err]: wrong img path.')
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,
                     (int(img.shape[1] * 0.3),
                      int(img.shape[0] * 0.3)),
                     cv2.INTER_CUBIC)
    cv2.imshow('src', img)

    # kernel_x = np.array([-1, 0, 1])
    # kernel_y = np.transpose(kernel_x)

    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    # kernel_x = np.array([[-1, 0, 1],
    #                      [-1, 0, 1],
    #                      [-1, 0, 1]])
    # kernel_y = np.array([[1, 1, 1],
    #                      [0, 0, 0],
    #                      [-1, -1, -1]])

    # sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, cv2.BORDER_DEFAULT)
    # sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, cv2.BORDER_DEFAULT)
    # sobel_x = cv2.convertScaleAbs(sobel_x)
    # sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_x = cv2.Sobel(img, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(img, -1, 0, 1, cv2.BORDER_DEFAULT)

    # 将x, y方向梯度结合起来
    sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # filter_x = cv2.filter2D(img, cv2.CV_16S, kernel_x, cv2.BORDER_DEFAULT)
    # filter_y = cv2.filter2D(img, cv2.CV_16S, kernel_y, cv2.BORDER_DEFAULT)
    # filter_x = cv2.convertScaleAbs(filter_x)
    # filter_y = cv2.convertScaleAbs(filter_y)

    filter_x = cv2.filter2D(img, -1, kernel_x, cv2.BORDER_DEFAULT)
    filter_y = cv2.filter2D(img, -1, kernel_y, cv2.BORDER_DEFAULT)

    # 测试二阶导数
    filter_xx = cv2.filter2D(filter_x, -1, kernel_x, cv2.BORDER_DEFAULT)
    filter_yy = cv2.filter2D(filter_y, -1, kernel_y, cv2.BORDER_DEFAULT)

    # to verify the sobel operator
    # for row in range(img.shape[0]):
    #     for col in range(img.shape[1]):
    #         if filter_x[row][col] != sobel_x[row][col]:
    #             print('=> filter_x and sobel_x not the same.')
    #         else:
    #             continue

    cv2.imshow('sobel_x', sobel_x)
    cv2.imshow('sobel_y', sobel_y)
    cv2.imshow('sobel_xy', sobel_xy)

    cv2.imshow('filter_x', filter_x)
    cv2.imshow('filter_y', filter_y)

    cv2.imshow('filter_xx', filter_xx)
    cv2.imshow('filter_yy', filter_yy)

    cv2.waitKey()

# ----------------------------------


def testFilter2D():
    img_path = 'c:/guide.jpg'
    # out_path = 'c:/test_result'

    if not os.path.isfile(img_path):
        print('[Err]: wrong img path.')
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,
                     (int(img.shape[1] * 0.3),
                      int(img.shape[0] * 0.3)),
                     cv2.INTER_CUBIC)

    cv2.imshow('src', img)
    # cv2.waitKey()

    kernel_x = np.array([1, -2, 1])  # sobel算子
    kernel_y = np.transpose(kernel_x)  # 卷积的可分离性
    kernel_xy = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

    filtered_x = cv2.filter2D(img, -1, kernel_x, cv2.BORDER_DEFAULT)
    filtered_y = cv2.filter2D(img, -1, kernel_y, cv2.BORDER_DEFAULT)

    filtered_xx = cv2.filter2D(filtered_x, -1, kernel_x, cv2.BORDER_DEFAULT)
    filtered_yy = cv2.filter2D(filtered_y, -1, kernel_y, cv2.BORDER_DEFAULT)

    filtered_x_y = cv2.filter2D(filtered_x, -1, kernel_y, cv2.BORDER_DEFAULT)
    filtered_y_x = cv2.filter2D(filtered_y, -1, kernel_x, cv2.BORDER_DEFAULT)
    filtered_xy = cv2.filter2D(img, -1, kernel_xy, cv2.BORDER_DEFAULT)

    sobel_x = cv2.Sobel(img, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(img, -1, 0, 1, cv2.BORDER_DEFAULT)

    sobel_xx = cv2.Sobel(sobel_x, -1, 1, 0, cv2.BORDER_DEFAULT)
    sobel_yy = cv2.Sobel(sobel_y, -1, 0, 1, cv2.BORDER_DEFAULT)

    sobel_x_y = cv2.Sobel(sobel_x, -1, 0, 1, cv2.BORDER_DEFAULT)

    laplace = cv2.Laplacian(img, -1, cv2.BORDER_DEFAULT)

    cv2.imshow('filter x', filtered_x)
    cv2.imshow('filter y', filtered_y)
    cv2.imshow('filter x_y', filtered_x_y)
    cv2.imshow('filter y_x', filtered_y_x)

    cv2.imshow('filter xy', filtered_xy)
    cv2.imshow('laplacian', laplace)

    cv2.imshow('filter_xx', filtered_xx)
    cv2.imshow('filter_yy', filtered_yy)

    cv2.imshow('sobel_x', sobel_x)
    cv2.imshow('sobel_y', sobel_y)
    cv2.imshow('sobel_xx', sobel_xx)
    cv2.imshow('sobel_yy', sobel_yy)

    cv2.imshow('sobel_x_y', sobel_x_y)

    cv2.waitKey()


if __name__ == '__main__':
    # testFilter2D()

    # testSobel()

    # testContour()

    # TestPatchMatchOptimize()

    Test2()

    print('=> Test done.')
