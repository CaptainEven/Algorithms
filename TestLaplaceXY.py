# coding:utf-8

import os
import numpy as np
import cv2


# ------------- PatchMatch optimization of depth map
def TestPatchMatchOptimize():
    src_path = 'C:/MyColMap/colmap-dev/workspace/SrcMVS/dense_output/images/IMG_2350.JPG'
    depthMap_path = 'C:/MyColMap/colmap-dev/workspace/result/depth_maps/IMG_2350.JPG.geometric.bin.jpg'

    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    depth_map = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)

    depth_h, depth_w = depth_map.shape
    src_resize = cv2.resize(src,
                            (depth_w, depth_h),
                            cv2.INTER_CUBIC)

    cv2.imshow('src_resize', src_resize)
    cv2.waitKey()
    cv2.imwrite('c:/src.jpg', src_resize)

    assert src_resize.shape == depth_map.shape
    



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

    testSobel()

    # TestPatchMatchOptimize()

    print('=> Test done.')
