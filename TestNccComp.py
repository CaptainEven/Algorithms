# encoding=utf-8

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def GetHomography(img1, img2, MIN=10):
    """
    """
    if img1 is None or img2 is None:
        print("[Err]: empty image.")
        return

    detector = cv2.xfeatures2d.SIFT_create()  # choose sift or surf
    # detector = cv2.xfeatures2d.SURF_create(10000, nOctaves=4, extended=False, upright=True)
    kp1, descrip1 = detector.detectAndCompute(img1, None)
    kp2, descrip2 = detector.detectAndCompute(img2, None)

    # ----------------- match features and find good match
    FLANN_INDEX_KDTREE = 0

    # Randomizedk-d tree, number of trees
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=50)  # specify the number of recursion

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    match = flann.knnMatch(descrip1, descrip2, k=2)

    # find good matches
    good = []
    for i, (m, n) in enumerate(match):
        if(m.distance < 0.5 * n.distance):
            good.append(m)

    # -----------------
    if len(good) > MIN:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # warp img1 to img2
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # print("Homography:\n", H)
        return H
    else:
        return None


def GetNCC(patch1, patch2):
    """
    Using normalized cross correlation as image similarity
    """
    if patch1.shape != patch2.shape:
        print("patch1.shape: ", patch1.shape, "patch2.shape: ", patch2.shape)
        # assert patch1.shape == patch2.shape
        return -2.0

    N = patch1.shape[0] * patch1.shape[1]

    data_1, data_2 = patch1.reshape(1, N), patch2.reshape(1, N)
    data_1, data_2 = data_1.astype(float), data_2.astype(float)

    # data standarddization
    u_1, std_1 = float(np.mean(data_1)), float(np.std(data_1))
    u_2, std_2 = float(np.mean(data_2)), float(np.std(data_2))

    data_1 -= u_1
    data_1 /= std_1
    data_2 -= u_2
    data_2 /= std_2

    # print('=> data_1 mean: %.3f' % (np.mean(data_1)))
    # print('=> data_2 mean: %.3f' % (np.mean(data_2)))

    # print('=> data_1 std: %.3f' % (np.std(data_1)))
    # print('=> data_2 std: %.3f' % (np.std(data_2)))

    # print('=> data_1 L2 norm: ', np.linalg.norm(data_1, ord=2))
    # print('=> data_2 L2 norm: ', np.linalg.norm(data_2, ord=2))

    # calculate the angle cosine
    ncc = np.dot(data_1, data_2.T) / \
        ((np.linalg.norm(data_1, ord=2)
          * np.linalg.norm(data_2, ord=2)))

    return ncc


def GetBWNCC(patch1, patch2, sigma_c=2.0, sigma_s=2.0):
    """
    计算双边权重NCC
    """
    # 计算Patch中各个像素的权重
    patch1_weights = np.zeros(patch1.shape)
    patch2_weights = np.zeros(patch2.shape)

    y_range1, x_range1 = np.arange(patch1.shape[0]), np.arange(patch1.shape[1])
    y_gird1, x_grid1 = np.meshgrid(y_range1, x_range1)
    center_x1 = (patch1.shape[1] - 1) // 2
    center_y1 = (patch1.shape[0] - 1) // 2
    space_dist1 = np.sqrt((center_x1 - x_grid1)**2 + (center_y1 - y_gird1)**2)
    color_dist1 = (patch1[y_gird1, x_grid1] - patch1[center_y1, center_x1])
    patch1_weights[y_gird1, x_grid1] = np.exp(-0.5 * color_dist1**2 / sigma_c**2) \
        * np.exp(-0.5 * space_dist1**2 / sigma_s**2)

    y_range2, x_range2 = np.arange(patch2.shape[0]), np.arange(patch2.shape[1])
    y_gird2, x_grid2 = np.meshgrid(y_range2, x_range2)
    center_x2 = (patch2.shape[1] - 1) // 2
    center_y2 = (patch2.shape[0] - 1) // 2
    space_dist2 = np.sqrt((center_x2 - x_grid2)**2 + (center_y2 - y_gird2)**2)
    color_dist2 = (patch2[y_gird2, x_grid2] - patch2[center_y2, center_x2])
    patch2_weights[y_gird2, x_grid2] = np.exp(-0.5 * color_dist2**2 / sigma_c**2) \
        * np.exp(-0.5 * space_dist2**2 / sigma_s**2)

    # 计算加权patch值
    patch1 = patch1 * patch1_weights
    patch2 = patch2 * patch2_weights

    return GetNCC(patch1, patch2)


def GetSSIM(patch1, patch2, k_1=0.01, k_2=0.03, L=255):
    """
    structure similarity between 2 imgages
    """
    assert patch1.shape == patch2.shape

    N = patch1.shape[0] * patch1.shape[1]
    u_1, u_2 = np.mean(patch1), np.mean(patch2)
    std_1, std_2 = np.std(patch1), np.std(patch2)

    c_1, c_2 = (k_1 * L)**2, (k_2 * L)**2

    data_1, data_2 = patch1.reshape(1, N), patch2.reshape(1, N)
    cov = np.cov(data_1, data_2)
    cov_12 = cov[0][1]

    ssim = (2 * u_1 * u_2 + c_1) * (2 * cov_12 + c_2)
    denom = (u_1**2 + u_2**2 + c_1) * (std_1**2 + std_2**2 + c_2)
    ssim /= denom

    return ssim


def GetSSD(patch1, patch2):
    """
    sum of squared error
    """
    assert patch1.shape == patch2.shape

    diff = patch1 - patch2
    sd = np.array(diff)**2

    return np.sum(sd)


def GetNMI(patch1, patch2, num_bins=20):
    """
    计算两个patch的归一化互信息
    基于直方图
    """
    assert patch1.shape == patch2.shape

    eps = 1.4e-45

    # 计算直方图,有了直方图其实可以计算两个概率分布的巴氏距离
    histgram1, IDLIst1 = GetHistgram(patch1, num_bins)
    histgram2, IDLIst2 = GetHistgram(patch2, num_bins)
    # print("Histgram1:\n", histgram1)
    # print("Histgram2:\n", histgram2)
    IDLIst1 = IDLIst1.astype(np.int)
    IDLIst2 = IDLIst2.astype(np.int)

    The1IDs = set(IDLIst1)
    The2IDs = set(IDLIst2)

    if The2IDs != The2IDs:
        return eps

    # 计算互信息
    Size = float(np.size(patch1))
    MI = 0.0
    for id1 in The1IDs:
        for id2 in The2IDs:
            id1_inds = np.where(IDLIst1 == id1)
            id2_inds = np.where(IDLIst2 == id2)

            id12_inds = np.intersect1d(id1_inds, id2_inds)
            # print(id12_inds)

            p1 = float(len(id1_inds[0])) / Size
            p2 = float(len(id2_inds[0])) / Size
            p12 = float(len(id12_inds)) / Size
            MI += p12*math.log2(p12 / (p1 * p2) + eps)

    # print("MI: %.3f" % MI)

    # 标准化互信息
    H1 = 0.0
    for id1 in The1IDs:
        id1OccurCount = float(len(np.where(IDLIst1 == id1)[0]))
        Prob = id1OccurCount / Size
        H1 = H1 - Prob*math.log2(Prob+eps)
    H2 = 0.0
    for id2 in The2IDs:
        id2Count = float(len(np.where(IDLIst2 == id2)[0]))
        Prob = id2Count/Size
        H2 = H2 - Prob*math.log2(Prob+eps)

    # print("H1: %.3f" % (H1))
    # print("H2: %.3f" % (H2))
    NMI = (MI + MI) / (H1 + H2)
    # print("NMI: %.3f" % NMI)

    return NMI


def GetHistgram(patch, num_bins):
    """
    输入patch，输出histgram
    """
    bin_vals = np.linspace(np.min(patch), np.max(patch), num_bins + 1)
    # print(bin_vals)

    histgrams = np.zeros(num_bins)
    ids = np.zeros(np.size(patch))

    for k, val in enumerate(np.nditer(patch, order="C")):
        for i in range(num_bins):  # [0, num_bins-1]
            if val >= bin_vals[i] and val < bin_vals[i + 1]:
                histgrams[i] += 1.0
                ids[k] = i
            elif val == bin_vals[-1]:
                histgrams[-1] += 1.0
                ids[k] = num_bins - 1

    # print(histgrams)
    histgrams /= float(patch.size)
    histgrams /= np.sum(histgrams)

    # print("sum(histgram): ", np.sum(histgrams))
    return histgrams, ids

def GetBDist(patch1, patch2, num_bins=10):
    """
    计算直方图, 然后计算巴氏距离
    """

    assert patch1.shape == patch2.shape

    # 计算直方图,有了直方图其实可以计算两个概率分布的巴氏距离
    histgram1, IDLIst1 = GetHistgram(patch1, num_bins)
    histgram2, IDLIst2 = GetHistgram(patch2, num_bins)

    # p = np.array([0.65, 0.25, 0.07, 0.03])
    # q = np.array([0.62, 0.26, 0.10, 0.02])

    BC = np.sum(np.sqrt(histgram1 * histgram2))

    # 巴氏距离：
    b = -np.log(BC)
    # print("b: %.3f" % b)
    return 1.0 - b


def Test(TestNum=10, start=300, step=50, ROI_LEN=50, WIN_SIZE=5):
    """
    Test Ncc comparision between original and enhanced image pairs
    """
    # PATH_PREFIX = "D:/ETH3D/"
    # img1_path = PATH_PREFIX + \
    #     "multi_view_training_dslr_undistorted/courtyard/images/DSC_0286.JPG"
    # img2_path = PATH_PREFIX + \
    #     "multi_view_training_dslr_undistorted/courtyard/images/DSC_0287.JPG"
    img1_path = "./SRC1.jpg"
    img2_path = "./SRC2.jpg"

    if not (os.path.isfile(img1_path) and os.path.isfile(img2_path)):
        print("[Err]: invalid file path.")
        return

    def GetMeanNCC(img1, img2, CENTER, NCC_WIN_SIZE):
        """
        """
        H = GetHomography(img1, img2)
        if H is None:
            print("[Warning]: Homography is None.")
            return

        # NCC window size and radius
        RADIUS = int((NCC_WIN_SIZE - 1) / 2)

        ROI = img1[CENTER[1]: CENTER[1] + ROI_LEN,  # y
                   CENTER[0]: CENTER[0] + ROI_LEN]  # x
        # print("ROI shape: ", ROI.shape)

        # ------------------- 遍历roi
        NCCSum = 0.0
        ROI_H, ROI_W = ROI.shape

        # PATCH_PAIR = np.zeros((NCC_WIN_SIZE, NCC_WIN_SIZE + 10 + NCC_WIN_SIZE))
        for y in range(ROI_H):  # rows
            for x in range(ROI_W):  # cols
                src_x = CENTER[0] + x
                src_y = CENTER[1] + y
                patch_1 = img1[src_y - RADIUS: src_y + RADIUS + 1,
                               src_x - RADIUS: src_x + RADIUS + 1]

                warp_coord = np.matmul(H, np.array(
                    [src_x, src_y, 1]))  # x, y, 1
                # print(warp_coord)
                dst_coord = [warp_coord[0] / warp_coord[2],  # x
                             warp_coord[1] / warp_coord[2]]  # y
                x_warp = int(dst_coord[0] + 0.5)  # x
                y_warp = int(dst_coord[1] + 0.5)  # y
                patch_2 = img2[y_warp - RADIUS: y_warp + RADIUS + 1,  # y
                               x_warp - RADIUS: x_warp + RADIUS + 1]  # x

                ncc = float(GetNCC(patch_1, patch_2))
                if ncc == -2.0:
                    print("patch shape not the same.")
                    break
                # print("%.3f" %ncc)

                NCCSum += ncc

        NCCMean = NCCSum / float(ROI_LEN * ROI_LEN)
        return NCCMean

    for i in range(TestNum):
        # x, y
        CENTER = [start + i * step, start + i * step]
        print("Center: [%d, %d]" % (CENTER[0], CENTER[1]))

        img1_path = img1_path.replace("ENH1", "SRC1")
        img2_path = img2_path.replace("ENH2", "SRC2")
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        NCCMean = GetMeanNCC(img1, img2, CENTER, WIN_SIZE)
        print("Mean Ncc of origin:  %.3f" % (NCCMean))

        img1_path = img1_path.replace("SRC1", "ENH1")
        img2_path = img2_path.replace("SRC2", "ENH2")
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        NCCMean = GetMeanNCC(img1, img2, CENTER, WIN_SIZE)
        print("Mean Ncc of enhance: %.3f\n" % (NCCMean))


def Test2(TestNum=10, start=310, step=3, ROI_LEN=7, WIN_SIZE=15):
    """
    Test Ncc comparision between original and enhanced image pairs
    """
    img1_path = "./Left_SRC.jpg"
    img2_path = "./Right_SRC.jpg"

    global H_  # 统一H以便比较
    H_ = None

    if not (os.path.isfile(img1_path) and os.path.isfile(img2_path)):
        print("[Err]: invalid file path.")
        return

    def GetMeanNCC(img1, img2, CENTER, NCC_WIN_SIZE, iter_i):
        """
        """
        global H_
        if i == 0:
            H = GetHomography(img1, img2)
            H_ = H
        if H_ is None:
            print("[Warning]: Homography is None.")
            return

        # H = GetHomography(img1, img2)
        # if H is None:
        #     print("[Warning]: Homography is None.")
        #     return

        # NCC window size and radius
        RADIUS = int((NCC_WIN_SIZE - 1) / 2)

        ROI = img1[CENTER[1]: CENTER[1] + ROI_LEN,  # y
                   CENTER[0]: CENTER[0] + ROI_LEN]  # x
        # print("ROI shape: ", ROI.shape)

        # ------------------- 遍历roi
        SimiSum = 0.0
        ROI_H, ROI_W = ROI.shape

        # PATCH_PAIR = np.zeros((NCC_WIN_SIZE, NCC_WIN_SIZE + 10 + NCC_WIN_SIZE))
        for y in range(ROI_H):  # rows
            for x in range(ROI_W):  # cols
                src_x = CENTER[0] + x
                src_y = CENTER[1] + y
                patch_1 = img1[src_y - RADIUS: src_y + RADIUS + 1,
                               src_x - RADIUS: src_x + RADIUS + 1]

                # ------ warping coordinate
                warp_coord = np.matmul(H_, np.array(
                    [src_x, src_y, 1]).T)  # x, y, 1
                # print(warp_coord)
                dst_coord = [warp_coord[0] / warp_coord[2],  # x
                             warp_coord[1] / warp_coord[2]]  # y
                x_warp = int(dst_coord[0] + 0.5)  # x
                y_warp = int(dst_coord[1] + 0.5)  # y
                patch_2 = img2[y_warp - RADIUS: y_warp + RADIUS + 1,  # y
                               x_warp - RADIUS: x_warp + RADIUS + 1]  # x

                # calculate NCC using bilateral weights
                simi = float(GetBWNCC(patch_1, patch_2))
                # simi = float(GetNMI(patch_1, patch_2, num_bins=10))
                # simi = float(GetBDist(patch_1, patch_2, num_bins=10))
                # simi = float(GetSSIM(patch_1, patch_2))
                # simi = float(GetSSD(patch_1, patch_2))

                if simi == -2.0:
                    print("patch shape not the same.")
                    break
                # print("%.3f" %simi)

                SimiSum += simi

        SimiMean = SimiSum / float(ROI_LEN * ROI_LEN)
        return SimiMean

    for i in range(TestNum):
        # x, y
        CENTER = [start + i * step, start + i * step]
        print("Center: [%d, %d]" % (CENTER[0], CENTER[1]))

        img1_path = img1_path.replace("ENH", "SRC")
        img2_path = img2_path.replace("ENH", "SRC")
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图(单通道)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        NCCMean = GetMeanNCC(img1, img2, CENTER, WIN_SIZE, i)
        print("Mean win%d similarity of origin:  %.3f" % (WIN_SIZE, NCCMean))

        img1_path = img1_path.replace("SRC", "ENH")  # 读取灰度图
        img2_path = img2_path.replace("SRC", "ENH")
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        NCCMean = GetMeanNCC(img1, img2, CENTER, WIN_SIZE, i)
        print("Mean win%d similarity of enhance: %.3f\n" % (WIN_SIZE, NCCMean))


def DrawSimilarity(f_path):
    """
    绘制NCC对比曲线: win5和win11
    """
    if not os.path.isfile(f_path):
        print("[Err]: invalid file path.")
        return

    # with open(f_path, "r", encoding="utf-8") as f_h:
    #     for line in f_h.readlines():
    #         if line.strip() == "Win5":
    #             print(line)
    #         elif line.strip == "Win11":
    #             print(line)
    #         else:
    #             continue

    # ori = [0.730, 0.725, 0.729, 0.781, 0.738, 0.775, 0.754, 0.725, 0.783, 0.731]
    # enh = [0.639, 0.617, 0.642, 0.661, 0.631, 0.660, 0.665, 0.655, 0.692, 0.648]
    # x = np.linspace(0, 9, num=10)
    # print(x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, ori, "c.-")
    # plt.plot(x, enh, "m.-")
    # plt.legend(labels=["Orignin", "Enhance"])
    # plt.xticks(x)
    # plt.title("Coutyard(texture)Win5")
    # plt.xlabel("Positions")
    # plt.ylabel("Ncc")
    # # plt.show()
    # plt.savefig("./Coutyard(texture)Win5.png")

    # ori = [0.861, 0.830, 0.828, 0.866, 0.825, 0.873, 0.837, 0.817, 0.858, 0.820]
    # enh = [0.736, 0.696, 0.718, 0.729, 0.681, 0.750, 0.730, 0.720, 0.754, 0.715]
    # x = np.linspace(0, 9, num=10)
    # print(x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, ori, "c.-")
    # plt.plot(x, enh, "m.-")
    # plt.legend(labels=["Orignin", "Enhance"])
    # plt.xticks(x)
    # plt.title("Coutyard(texture)Win11")
    # plt.xlabel("Positions")
    # plt.ylabel("Ncc")
    # # plt.show()
    # plt.savefig("./Coutyard(texture)Win11.png")

    # # ------- Office
    # ori = [0.773, 0.759, 0.773, 0.779, 0.779, 0.778, 0.779, 0.777, 0.784, 0.762]
    # enh = [0.703, 0.677, 0.682, 0.692, 0.684, 0.676, 0.679, 0.676, 0.690, 0.683]
    # x = np.linspace(0, 9, num=10)
    # print(x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, ori, "c.-")
    # plt.plot(x, enh, "m.-")
    # plt.legend(labels=["Orignin", "Enhance"])
    # plt.xticks(x)
    # plt.title("Office(textureless)Win5")
    # plt.xlabel("Positions")
    # plt.ylabel("Ncc")
    # # plt.show()
    # plt.savefig("./Office(textureless)Win5.png")

    # ori = [0.931, 0.926, 0.930, 0.935, 0.935, 0.936, 0.933, 0.935, 0.938, 0.907]
    # enh = [0.866, 0.853, 0.853, 0.863, 0.856, 0.849, 0.852, 0.851, 0.857, 0.850]
    # x = np.linspace(0, 9, num=10)
    # print(x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, ori, "c.-")
    # plt.plot(x, enh, "m.-")
    # plt.legend(labels=["Orignin", "Enhance"])
    # plt.xticks(x)
    # plt.title("Office(textureless)Win11")
    # plt.xlabel("Positions")
    # plt.ylabel("Ncc")
    # # plt.show()
    # plt.savefig("./Office(textureless)Win11.png")

    ori = [0.765, 0.757, 0.762, 0.782, 0.775,
           0.783, 0.782, 0.780, 0.779, 0.784]
    enh = [0.798, 0.786, 0.803, 0.820, 0.805,
           0.800, 0.814, 0.816, 0.823, 0.788]
    x = np.linspace(0, 9, num=10)
    print(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, ori, "c.-")
    plt.plot(x, enh, "m.-")
    plt.legend(labels=["Orignin", "Enhance"])
    plt.xticks(x)
    plt.title("Office(textureless)Win5_new")
    plt.xlabel("Positions")
    plt.ylabel("Ncc")
    # plt.show()
    plt.savefig("./Office(textureless)Win5_new.png")


if __name__ == "__main__":
    Test2()
    # DrawSimilarity(f_path="./office_ncc_test_win5_new.txt")

    print("Done.")
