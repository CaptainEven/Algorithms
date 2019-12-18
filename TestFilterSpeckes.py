# encoding=utf-8

import os
import copy
import cv2
import numpy as np


def FilterSpeckles(root, suffix):
    """
    batch processing
    """
    if not os.path.isdir(root):
        print("[Err]: invalid root.")
        return

    for f_name in os.listdir(root):
        if not f_name.endswith(suffix):
            continue

        f_path = root + "/" + f_name
        img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("[Err]: empty image.")
            return

        if len(img.shape) == 2:
            H, W = img.shape
        elif len(img.shape) == 3:
            H, W, N = img.shape
        maxSpeckleSize = int((H * W) / 810.0 + 0.5)
        maxDiff = 10

        # ------------- remove speckles
        out = copy.deepcopy(img)
        cv2.filterSpeckles(out, 0, maxSpeckleSize, maxDiff)
        # -------------

        # save output
        out_name = f_name.replace(".jpg", "_filter.jpg")
        out_path = root + "/" + out_name
        cv2.imwrite(out_path, out)
        print("%s exported" % (out_name))


def Test():
    """
    """
    dir = "E:/office/dense/stereo/depth_maps/dslr_images_undistorted"
    # dir = "F:/ETH3DResults/pipes/result/depth_maps"
    # img_path = dir + "/" + "DSC_0635.JPG.geometric.bin.jpg"
    img_path = dir + "/" + "DSC_0219.JPG.geometric.bin_win6.jpg"

    if not os.path.isfile(img_path):
        print("[Err]: invalid file path")
        return

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("[Err]: empty image.")
        return

    RATIO = 0.35

    # H, W = img.shape
    if len(img.shape) == 2:
        H, W = img.shape
    elif len(img.shape) == 3:
        H, W, N = img.shape

    img_rs = cv2.resize(img,
                        (int(W * RATIO), int(H * RATIO)),
                        cv2.INTER_LINEAR)
    cv2.imshow("img", img_rs)

    maxSpeckleSize = int((H * W) / 810.0 + 0.5)
    maxDiff = 10

    # ------------- remove speckles
    out = copy.deepcopy(img)
    cv2.filterSpeckles(out, 0, maxSpeckleSize, maxDiff)  # uint8怪不得可以处理...
    # -------------

    out_rs = cv2.resize(out,
                        (int(W * RATIO), int(H * RATIO)),
                        cv2.INTER_LINEAR)
    cv2.imshow("filtered", out_rs)
    cv2.waitKey()


if __name__ == "__main__":
    # Test()
    FilterSpeckles(root="e:/office/dense/stereo/depth_maps/dslr_images_undistorted",
                   suffix="win6.jpg")
    print("Done")
