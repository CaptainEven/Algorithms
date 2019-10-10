# coding=utf-8

import numpy as np
from PIL import Image
import cv2
import time


def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2

    patch_a = A_padding[a[0]: a[0]+p_size, a[1]: a[1]+p_size, :]
    patch_b = B[b[0]-p: b[0]+p+1, b[1]-p: b[1]+p+1, :]

    temp = patch_b - patch_a

    # non-nan mask
    mask = np.where(~np.isnan(temp))  # np.where(~np.isnan(temp))

    dist = np.sqrt(np.sum(np.square(temp[mask])) / temp[mask].size)

    # only calculate the image area, padding area is not considered
    # num = np.sum(1 - np.int32(np.isnan(temp)))
    # dist = np.sqrt(np.sum(np.square(np.nan_to_num(temp))) / num)  # / num

    return dist


def reconstruction(field_offsets, A, B):
    """
    A: input image
    B: reference
    TODO: speed up for loop with numpy APIs
    """

    A_h = np.size(A, 0)
    A_w = np.size(A, 1)

    temp = np.zeros_like(A)

    # reconstruction each pixel in A using pixel from reference image
    for y in range(A_h):
        for x in range(A_w):
            temp[y, x, :] = B[field_offsets[y, x][0], field_offsets[y, x][1], :]
    
    # -------- show for comparison
    cv2.imshow('input', A[:, :, ::-1])
    cv2.imshow('reference', B[:, :, ::-1])
    cv2.imshow('reconstructed', temp[:, :, ::-1])
    cv2.waitKey()
    # --------

    # Image.fromarray(temp).show()


def initialization(A, B, p_size):
    """
    A: img
    B: ref
    """

    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)

    pad = p_size // 2
    random_B_y = np.random.randint(pad, B_h-pad, [A_h, A_w])
    random_B_x = np.random.randint(pad, B_w-pad, [A_h, A_w])

    # padding for input image A
    A_padding = np.ones([A_h+pad*2, A_w+pad*2, 3]) * np.nan
    A_padding[pad:A_h+pad, pad:A_w+pad, :] = A
    
    # init offset field
    f_offsets = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])

    for y in range(A_h):
        for x in range(A_w):
            a = np.array([y, x])
            b = np.array([random_B_y[y, x], random_B_x[y, x]], dtype=np.int32)

            f_offsets[y, x] = b  # (y, x)
            dist[y, x] = cal_distance(a, b, A_padding, B, p_size)

    return f_offsets, dist, A_padding


def propagation(f_offsets, a, dist, A_padding, B, p_size, is_even):
    """
    """

    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    y = a[0]
    x = a[1]

    d_current = dist[y, x]
    if is_even:  # left-up
        d_up = dist[max(y - 1, 0), x]
        d_left = dist[y, max(x - 1, 0)]

        idx = np.argmin(np.array([d_current, d_up, d_left]))

        if idx == 1:
            f_offsets[y, x] = f_offsets[max(y - 1, 0), x]
            dist[y, x] = cal_distance(a, f_offsets[y, x], A_padding, B, p_size)
        if idx == 2:
            f_offsets[y, x] = f_offsets[y, max(x - 1, 0)]
            dist[y, x] = cal_distance(a, f_offsets[y, x], A_padding, B, p_size)
    else:  # right-down
        d_down = dist[min(y + 1, A_h-1), x]
        d_right = dist[y, min(x + 1, A_w-1)]

        idx = np.argmin(np.array([d_current, d_down, d_right]))

        if idx == 1:
            f_offsets[y, x] = f_offsets[min(y + 1, A_h-1), x]
            dist[y, x] = cal_distance(a, f_offsets[y, x], A_padding, B, p_size)
        if idx == 2:
            f_offsets[y, x] = f_offsets[y, min(x + 1, A_w-1)]
            dist[y, x] = cal_distance(a, f_offsets[y, x], A_padding, B, p_size)


def random_search(f_offsets, a, dist, A_padding, B, p_size, alpha=0.8):
    """
    """

    y = a[0]
    x = a[1]

    B_h = np.size(B, 0)
    B_w = np.size(B, 1)

    p = p_size // 2
    i = 0  # hyper-parameter

    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i

    b_y = f_offsets[y, x][0]
    b_x = f_offsets[y, x][1]
    
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_y - search_h, p)  # set min to p(patch radius)
        search_max_r = min(b_y + search_h, B_h - p)
        random_b_y = np.random.randint(search_min_r, search_max_r)

        search_min_c = max(b_x - search_w, p)
        search_max_c = min(b_x + search_w, B_w - p)
        random_b_x = np.random.randint(search_min_c, search_max_c)

        b = np.array([random_b_y, random_b_x])  # new offset in B
        d = cal_distance(a, b, A_padding, B, p_size)

        if d < dist[y, x]:
            dist[y, x] = d
            f_offsets[y, x] = b

        # update searching window size
        i += 1
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i


def ANNFs(img, ref, p_size, itr):
    """
    approximate nearest neighbor fields
    """

    A_h = np.size(img, 0)
    A_w = np.size(img, 1)

    f_offsets, dist, img_padding = initialization(img, ref, p_size)

    for itr in range(1, itr+1):
        if itr % 2 == 0:  # even iterations
            for y in range(A_h - 1, -1, -1):
                for x in range(A_w - 1, -1, -1):
                    a = np.array([y, x])

                    propagation(f_offsets, a, dist, img_padding, ref, p_size, True)
                    random_search(f_offsets, a, dist, img_padding, ref, p_size)
        else:  # odd iterations
            for y in range(A_h):
                for x in range(A_w):
                    a = np.array([y, x])

                    propagation(f_offsets, a, dist, img_padding, ref, p_size, False)
                    random_search(f_offsets, a, dist, img_padding, ref, p_size)
        print("=> iteration: %d" % (itr))

    return f_offsets


if __name__ == "__main__":
    img = np.array(Image.open('./cup_a.jpg'))
    ref = np.array(Image.open('./cup_b.jpg'))

    p_size = 3
    itr = 5

    start = time.time()

    # ------------------------------------------
    field_offsets = ANNFs(img, ref, p_size, itr)
    # ------------------------------------------    
    end = time.time()
    
    print('=> ANNFs run for %d secs.' %(end - start))

    reconstruction(field_offsets, img, ref)
