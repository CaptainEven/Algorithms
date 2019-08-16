# coding: utf-8

import cv2
import numpy as np
from scipy import misc
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction import image
import time
from PatchMatch import *

# This function builds an image of patches. Given an image im_in, an
# index idx characterizing a pixel position in im_in, a vector of
# offsets for that pixel and the set of patches extracted from im_in,
# the output represents the subset of patches given by idx + offsets[i], i=0...len(offsets)


def patch_cloud(im_in, idx, offsets, w, patches):
    [M, N] = im_in.shape
    m = 2*w+1
    im2 = np.zeros(im_in.shape, dtype=float)
    [i0, j0] = idx1d_to_idx2d(int(idx), M-2*w, N-2*w)
    im2[i0: i0+m, j0: j0+m] = patches[idx]
    for of in offsets:
        idx1 = int(idx + of)
        [i1, j1] = idx1d_to_idx2d(idx1, M-2*w, N-2*w)
        im2[i1: i1+m, j1: j1+m] = patches[idx1]
    return im2


def test_patch_match():
    t0 = time.time()

    im_name = 'koala.png'
    # im_name = 'cup_a.jpg'

    # read in gray-scale image
    im = misc.imread(im_name, 'L')
    im = 255.0 * im / im.max()

    # crop image to work with a smaller one
    i0 = 300
    j0 = 400
    w_im = 100
    im = im[i0-w_im: i0+w_im+1, j0-w_im: j0+w_im+1]

    # ROI = im.astype('uint8')
    # cv2.imwrite('./ROI.png', ROI)

    # cv2.imshow('ROI', ROI)
    # cv2.waitKey()

    [M, N] = im.shape
    n = M*N

    ####################################################################
    # FIND APPROXIMATED NEAREST NEIGHBOURS WITH GENERALIZED PATCHMATCH #
    ####################################################################
    w_patch = 2  # 2

    # side of a patch; a patch is then m_patch x m_patch
    m_patch = 2*w_patch+1

    # Number of nearest neighbours
    knn = 5

    t0 = time.time()

    offsets, weights = patch_match(im, m_patch, knn)

    t1 = time.time()
    delta_t = t1 - t0
    print("=> Execution took " + str(delta_t) + 's\n')

    ##################################################
    # Show the approximated knn nearest patches
    ##################################################
    im_bis = np.pad(im, [w_patch, w_patch], 'symmetric')
    patches = image.extract_patches_2d(im_bis.T, (m_patch, m_patch))

    # for i in range(n):
    #     patches[i] = patches[i].T

    patches = [patch.T for patch in patches]

    # make sure offsets are integers
    offsets = offsets.astype(int)

    # choose a pixel at random
    idx = int(np.floor((n-1) * np.random.rand()))
    idx = 19385

    # compute 2D coordinates from pixel's index
    [i0, j0] = idx1d_to_idx2d(int(idx), M, N)

    # Subset of offsets locating the knn approximated nearest neigbours
    offsets_for_pixel_idx = offsets[idx, ]

    # image showing the knn approximated nearest patches, 
    # including the reference patch
    im_patches = patch_cloud(im_bis,
                             idx,
                             offsets_for_pixel_idx,
                             w_patch,
                             patches)

    # cut off external w_patch wide stripes from im_patches 
    # to match the original image's size
    [Mbis, Nbis] = im_patches.shape
    im_patches = im_patches[w_patch:Mbis-w_patch, w_patch:Nbis-w_patch]

    # Mark the reference pixel with a white cross
    im[i0, j0] = 255
    im[min(i0+1, M-1), j0] = 255
    im[max(i0-1, 0), j0] = 255
    im[i0, min(j0+1, N-1)] = 255
    im[i0, max(j0-1, 0)] = 255

    plt.close('all')
    plt.figure()
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.figure()
    plt.imshow(im_patches, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    test_patch_match()
    print('=> Test done.')
