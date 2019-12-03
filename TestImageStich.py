# encoding:utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


MIN = 10
starttime = time.time()
img1 = cv2.imread('./img_1.png')  # query
img2 = cv2.imread('./img_2.png')  # train

# img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# surf = cv2.xfeatures2d.SURF_create(10000, nOctaves=4, extended=False, upright=True)
surf = cv2.xfeatures2d.SIFT_create()  # choose sift or surf
kp1, descrip1 = surf.detectAndCompute(img1, None)
kp2, descrip2 = surf.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
# Randomizedk-d tree, number of trees
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=50)  # specify the number of recursion

flann = cv2.FlannBasedMatcher(index_params, search_params)
match = flann.knnMatch(descrip1, descrip2, k=2)

# find good matches?
good = []
for i, (m, n) in enumerate(match):
    if(m.distance < 0.75 * n.distance):
        good.append(m)

if len(good) > MIN:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # warped_img2 = cv2.warpPerspective(img2,
    #                               np.linalg.inv(M),
    #                               (img1.shape[1] + img2.shape[1], img2.shape[0]))

    # warp img2 to img1
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    warped_img2 = cv2.warpPerspective(img2,
                                      M,
                                      (img1.shape[1] + img2.shape[1], img2.shape[0]))

    direct = warped_img2.copy()
    direct[0:img1.shape[0], 0:img1.shape[1]] = img1

    # warped_img2_rs = cv2.resize(warped_img2,
    #                             (int(warped_img2.shape[1] * 0.3),
    #                              int(warped_img2.shape[0] * 0.3)),
    #                             cv2.INTER_CUBIC)
    # cv2.imshow("Warped img2", warped_img2_rs)
    # cv2.waitKey()

    simple = time.time()

    # ----------------------------------------
    rows, cols, _ = img1.shape
    for x in range(cols):
        # print(warped_img2[:, 380])
        # print(warped_img2[:, 380].any())
        # nonzero_ids_y, nonzero_ins_x = np.nonzero(warped_img2[:, 380])
        # print(nonzero_ids_y.size)

        if img1[:, x].any() and warped_img2[:, x].any():  # 开始重叠的最左端
            left = x
            break
        
    for x in range(cols-1, 0, -1):
        if img1[:, x].any() and warped_img2[:, x].any():  # 重叠的最右一列
            right = x
            break

    # blending img1 and warped_img2
    res = np.zeros([rows, cols, 3], np.uint8)
    for y in range(rows):
        for x in range(cols):
            if not img1[y, x].any():  # 如果没有原图, 用旋转的填充
                res[y, x] = warped_img2[y, x]
            elif not warped_img2[y, x].any():
                res[y, x] = img1[y, x]
            else:
                srcImgLen = float(abs(x - left))
                testImgLen = float(abs(x - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                pixel = img1[y, x] * \
                    (1-alpha) + warped_img2[y, x] * alpha
                res[y, x] = np.clip(pixel, 0, 255)

    # put blending result back
    warped_img2[0:img1.shape[0], 0:img1.shape[1]] = res

    # ----------------------------------------

    final = time.time()

    # img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
    # plt.imshow(img3,), plt.show()
    img3 = cv2.resize(direct,
                      (int(direct.shape[1] * 0.4),
                       int(direct.shape[0] * 0.4)),
                      cv2.INTER_CUBIC)
    cv2.imshow("img3", img3)

    # img4 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB)
    # plt.imshow(img4,), plt.show()
    img4 = cv2.resize(warped_img2,
                      (int(warped_img2.shape[1] * 0.4),
                       int(warped_img2.shape[0] * 0.4)),
                      cv2.INTER_CUBIC)
    cv2.imshow("img4", img4)
    cv2.waitKey()

    print("simple stich cost %f" % (simple-starttime))
    print("\ntotal cost %f" % (final-starttime))

    cv2.imwrite("SimplePanorma.png", direct)
    cv2.imwrite("BestPanorma.png", warped_img2)
    print("Done.")

else:
    print("not enough matches!")
