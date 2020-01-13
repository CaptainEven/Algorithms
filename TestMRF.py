import cv2
import numpy as np
import copy


img = cv2.imread('./test.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片二值化，彩色图片该方法无法做分割
img = gray 
img_double = np.array(img, dtype=np.float64)

cluster_num = 4
max_iter = 200

label = np.random.randint(1, cluster_num + 1, size=img_double.shape)

iter = 0
f_u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_d = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
f_l = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3)
f_r = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(3, 3)
f_ul = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_ur = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_dl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3)
f_dr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(3, 3)

while iter < max_iter:
    iter += 1
    print(iter)

    label_u = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_u)
    label_d = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_d)
    label_l = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_l)
    label_r = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_r)
    label_ul = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ul)
    label_ur = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ur)
    label_dl = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dl)
    label_dr = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dr)

    m, n = label.shape    
    p_c = np.zeros((cluster_num, m, n))

    for i in range(cluster_num):
        label_i = (i+1) * np.ones((m, n))

        u_T = 1 * np.logical_not(label_i - label_u)
        d_T = 1 * np.logical_not(label_i - label_d)
        l_T = 1 * np.logical_not(label_i - label_l)
        r_T = 1 * np.logical_not(label_i - label_r)
        ul_T = 1 * np.logical_not(label_i - label_ul)
        ur_T = 1 * np.logical_not(label_i - label_ur)
        dl_T = 1 * np.logical_not(label_i - label_dl)
        dr_T = 1 * np.logical_not(label_i - label_dr)
        temp = u_T + d_T + l_T + r_T + ul_T + ur_T + dl_T + dr_T

        p_c[i, :] = (1.0/8.0) * temp

    p_c[p_c == 0] = 0.001
    mu = np.zeros((1, cluster_num))
    sigma = np.zeros((1, cluster_num))

    for i in range(cluster_num):
        index = np.where(label == (i+1))
        data_c = img[index]
        mu[0, i] = np.mean(data_c)
        sigma[0, i] = np.var(data_c)

    p_sc = np.zeros((cluster_num, m, n))
    one_a = np.ones((m, n))

    for j in range(cluster_num):
        MU = mu[0, j] * one_a
        p_sc[j, :] = (1.0/np.sqrt(2 * np.pi * sigma[0, j])) * \
            np.exp(-1. * ((img - MU)**2) / (2 * sigma[0, j]))

    X_out = np.log(p_c) + np.log(p_sc)
    label_c = X_out.reshape(cluster_num, m * n)
    label_c_t = label_c.T

    # 优化: 取极大似然概率对应的标签
    label_m = np.argmax(label_c_t, axis=1)

    # 由于上一步返回的是index下标，与label其实就差1，因此加上一个ones矩阵即可
    label_m = label_m + np.ones(label_m.shape)
    label = label_m.reshape(m, n)

# 可视化
label = label.astype(np.uint8)

for c in range(cluster_num):
    intensity = int(c / cluster_num * 255)
    inds_y, inds_x = np.where(label==c+1)
    label[inds_y, inds_x] = intensity

label_show = cv2.applyColorMap(label, cv2.COLORMAP_HOT)

cv2.imshow("Label", label_show)
cv2.waitKey()

# https://blog.csdn.net/weixin_37752934/article/details/83385853
