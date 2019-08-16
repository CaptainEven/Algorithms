# coding: utf-8

import numpy as np
# from numpy import *
from scipy.stats import norm
import math
from sklearn.feature_extraction import image
import time
import heapq


def dissimilarity_weights_(m):
    w = int((m - 1) / 2)
    v = range(-w, w + 1)
    [x, y] = np.meshgrid(v, v, indexing='xy')

    # x_norm = norm.pdf(x)
    # y_norm = norm.pdf(y)
    # x_sum = np.sum(x_norm, axis=1)
    # y_sum = np.sum(y_norm, axis=0)

    g = norm.pdf(x) * norm.pdf(y)

    # g_sum = np.sum(g)

    return g


def dissimilarity_weights(m):
    """
    2D gauss probability distribution
    """
    w = int((m - 1) * 0.5)

    y, x = np.ogrid[-w:w + 1, -w:w + 1]

    # std_x, std_y = np.std(x, axis=1), np.std(y, axis=0)

    probs = np.exp(-0.5 * (x**2 + y**2))
    prob_sum = np.sum(probs)

    # normalize to [0, 1], with sum to 1.0
    if prob_sum != 0.0:
        probs /= prob_sum

    # print('=> probs sum: ', np.sum(probs))

    return probs


def patches_dissimilarity(p1, p2, alpha, gauss_weights_2d):
    """
    patch similarity increase => l2_loss decrease
    => dissim increase
    """

    diff_l2 = (p1 - p2)**2
    weighted_d2 = gauss_weights_2d * diff_l2
    dissim = -alpha * weighted_d2.sum()

    return dissim


def idx1d_to_idx2d(idx, M, N):
    j = int(np.floor(idx / M))
    i = idx - j * M
    return [i, j]  # row, col


def idx2d_to_idx1d(i, j, M, N):
    idx = int(M * j + i)
    return idx


def try_improve_a_from_b_(idx_a, idx_b,
                          heap_a, heap_b,
                          patches,
                          M, N,
                          alpha,
                          gauss_weights_2d):
    """
    """

    k = len(heap_a)

    # patch around pixel idx_a
    p0 = patches[idx_a, :, :]

    # neighbor's nearest neighbors
    for nn in range(k):
        # compute the 2D offset corresponding idx_b's nn-est nearest neighbour
        offs_b = heap_b[nn][1]

        idx_d = int(idx_a + offs_b)

        idx_d = max(idx_d, 0)
        idx_d = min(idx_d, patches.shape[0] - 1)
        offs_b = idx_d - idx_a

        # patch around the new pixel to compare to idx_a
        p2 = patches[idx_d, :, :]

        # new weight
        w_b = patches_dissimilarity(p0, p2, alpha, gauss_weights_2d)

        if w_b > heap_a[0][0] and not ((w_b, offs_b) in heap_a):
            heapq.heapreplace(heap_a, (w_b, offs_b))

    return heap_a


def try_improve_a_from_b(idx_a, idx_b,
                         heap_a, heap_b,
                         patches,
                         M, N,
                         alpha,
                         gauss_weights_2d):
    """
    """

    k = len(heap_a)

    # patch around pixel idx_a
    p0 = patches[idx_a, :, :]
    p1 = patches[idx_b, :, :]  # neighbor

    # neighbor's nearest neighbors
    for nn in range(k):
        # compute the 2D offset corresponding idx_b's nn-est nearest neighbour
        offs_b = heap_b[nn][1]

        idx_d = int(idx_b + offs_b)

        # to prevent index is out of range
        idx_d = max(idx_d, 0)
        idx_d = min(idx_d, patches.shape[0] - 1)

        offs_b = idx_d - idx_b

        # patch around the new pixel to compare to idx_a
        p2 = patches[idx_d, :, :]

        # neighbor's weight
        w_b = patches_dissimilarity(p1, p2, alpha, gauss_weights_2d)

        # user min-heap to keep top-K
        if w_b > heap_a[0][0]:
            # update new weight(patch similarity) and new offset
            if idx_a != idx_b:
                w_a = patches_dissimilarity(p0, p2, alpha, gauss_weights_2d)
            else:
                w_a = w_b
            
            offs_a = idx_d - idx_a
            heapq.heapreplace(heap_a, (w_a, offs_a))

    return heap_a


def build_new_offsets(idx, heap_0, L, l, q, M, N):
    """
    """

    new_offsets = np.zeros(len(heap_0))

    for k in range(len(heap_0)):

        idx2 = int(idx) + int(heap_0[k][1])
        [i2, j2] = idx1d_to_idx2d(idx2, M, N)

        [u, v] = (L * l**q) * (2 * np.random.rand(2, ) - 1)
        [i3, j3] = np.array([i2, j2]) + np.array([u, v])
        i3, j3 = int(i3), int(j3)

        i3 = max(0, i3)
        i3 = min(M, i3)
        j3 = max(0, j3)
        j3 = min(N, j3)

        idx3 = idx2d_to_idx1d(i3, j3, M, N)
        new_offsets[k] = idx3 - idx

    return new_offsets


def search_around(idx,
                  heap_0,
                  patches,
                  M, N,
                  alpha,
                  gauss_weights_2d):
    """
    """

    L = 30
    l = 0.5
    q_range = np.arange(0, 3)

    # 3×5
    cand_offsets = np.empty((q_range.shape[0], len(heap_0)))

    for q in q_range:
        cand_offsets[q, ] = build_new_offsets(idx, heap_0, L, l, q, M, N)

    for c in range(3):

        heap_1 = []

        for k in range(0, len(heap_0)):  # weight between 2 patches inited to 0
            heap_1.append((0, cand_offsets[c, k]))

        new_heap_0 = try_improve_a_from_b(idx,
                                          idx,
                                          heap_0, heap_1,
                                          patches,
                                          M, N,
                                          alpha,
                                          gauss_weights_2d)

    return new_heap_0


def initialize_offsets(M, N, k):
    """
    """
    n = M * N  # 184*201: 36984
    # print((n-1) * np.random.rand(n ,k))
    offsets = np.floor((n-1)*np.random.rand(n, k)) - \
        np.tile(np.arange(0, n), (k, 1)).T
    offsets.astype(int)

    # print(offsets.shape)

    return offsets  # shape: 36984*5


def initialize_weights(patches, offsets, alpha, gauss_weights_2d):
    """
    """

    [n, k] = offsets.shape

    # the more similar the 2 patches are, the larger corresponding weight
    weights = np.ones([n, k])

    for i in range(n):  # total n patches(pixels)

        p0 = patches[i, :, :]

        for j in range(k):  # k nearest neighbor for each patch
            i2 = int(i + offsets[i, j])

            p2 = patches[i2, :, :]  # patches: 36984*5*5
            weights[i, j] = patches_dissimilarity(p0,
                                                  p2,
                                                  alpha,
                                                  gauss_weights_2d)

    return weights


def initialize_heap(offsets, weights):
    """
    """
    all_heaps = []
    [n, k] = offsets.shape

    for i in range(n):  # each pixel's heap(top k neighbors)

        h = []

        for j in range(k):
            heapq.heappush(h, (weights[i, j], offsets[i, j]))

        # sort heap
        # heapq.heapify(h)

        # verified: min heap(by default) sorted by weight
        # for a in range(k):
        #     b = heapq.heappop(h)
        #     print(b)
        # print('\n')

        all_heaps.append(h)

    return all_heaps


def propagate(iter_n,
              current_heaps,
              M, N,
              patches,
              alpha,
              gauss_weights_2d):
    """
    """

    if iter_n % 2:  # odd iterations
        J_range = range(1, N)
        I_range = range(1, M)
        pix_shift = -1
    else:
        J_range = np.arange(N - 2, -1, -1)
        I_range = np.arange(M - 2, -1, -1)
        pix_shift = 1

    for j0 in J_range:  # cols
        for i0 in I_range:  # rows
            # process each pixel

            idx_0 = idx2d_to_idx1d(i0, j0, M, N)  # current processing pixel
            idx_1 = idx2d_to_idx1d(i0 + pix_shift, j0, M, N)  # above pixel
            idx_2 = idx2d_to_idx1d(i0, j0 + pix_shift, M, N)  # left pixel

            heap_0 = current_heaps[idx_0]
            heap_1 = current_heaps[idx_1]
            heap_2 = current_heaps[idx_2]

            heap_0 = try_improve_a_from_b(idx_0, idx_1,
                                          heap_0, heap_1,
                                          patches,
                                          M, N,
                                          alpha,
                                          gauss_weights_2d)

            current_heaps[idx_0] = try_improve_a_from_b(idx_0, idx_2,
                                                        heap_0, heap_2,
                                                        patches,
                                                        M, N,
                                                        alpha,
                                                        gauss_weights_2d)

    return current_heaps


def random_search(current_heaps,
                  M, N,
                  patches,
                  alpha,
                  gauss_weights_2d):
    """
    """

    n = M * N
    new_heaps = []

    for idx in range(n):
        # process each pixel
        heap_0 = current_heaps[idx]

        new_heap_0 = search_around(idx,
                                   heap_0,
                                   patches,
                                   M, N,
                                   alpha,
                                   gauss_weights_2d)

        new_heaps.append(new_heap_0)

    return new_heaps


def heaps_to_weights_and_offsets(all_heaps):
    n = len(all_heaps)
    k = len(all_heaps[0])
    offsets = np.zeros((n, k), dtype=int)
    weights = np.zeros((n, k))
    for i in range(0, n):
        for j in range(0, k):
            weights[i, j] = all_heaps[i][j][0]
            offsets[i, j] = int(all_heaps[i][j][1])
    return offsets, weights


def patch_match(im, m, knn):
    """
    """

    print("Starting PatchMatch\n")

    w = int((m - 1) / 2)
    alpha = 0.01
    gauss_weights_2d = dissimilarity_weights(m)
    im_bis = np.pad(im, [w, w], 'symmetric')

    # M rows N cols
    [M, N] = im.shape
    n = M * N
    print('=> total %d patches(pxiels)' % (n))

    print("=> Exctracting patches...")
    tstart = time.time()

    # why transpose?
    patches = image.extract_patches_2d(image=im_bis.T, patch_size=(m, m))

    for i in range(n):
        patches[i] = patches[i].T

    tend = time.time()

    print(str(tend - tstart) + " s.\n")
    print("=> Init offsets...")

    tstart = time.time()

    offsets_init = initialize_offsets(M, N, knn)

    tend = time.time()

    print(str(tend - tstart) + " s.\n")

    print("=> Init weights...")
    tstart = time.time()

    weights_init = initialize_weights(patches,
                                      offsets_init,
                                      alpha,
                                      gauss_weights_2d)

    tend = time.time()
    print(str(tend - tstart) + " s.\n")

    print("=> Init max-heaps...")
    tstart = time.time()

    all_heaps = initialize_heap(offsets_init, weights_init)

    tend = time.time()
    print(str(tend - tstart) + " s.\n")

    print("=> First propagation...")
    tstart = time.time()

    new_heaps = propagate(1,
                          all_heaps,
                          M,
                          N,
                          patches,
                          alpha,
                          gauss_weights_2d)

    tend = time.time()
    print(str(tend - tstart) + " s.\n")

    print("=> Random search...")
    tstart = time.time()

    new_heaps = random_search(new_heaps,
                              M, N,
                              patches,
                              alpha,
                              gauss_weights_2d)

    tend = time.time()
    print(str(tend - tstart) + " s.\n")

    print("=> Second propagation...")
    tstart = time.time()

    new_heaps = propagate(2, new_heaps, M, N, patches, alpha, gauss_weights_2d)

    tend = time.time()
    print(str(tend - tstart) + " s.\n")

    print("=> Convert heaps to offsets and weights...")
    tstart = time.time()

    offsets, weights = heaps_to_weights_and_offsets(all_heaps)

    tend = time.time()
    print(str(tend - tstart) + " s.\n")

    return offsets, weights
