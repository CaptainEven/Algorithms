# coding: utf-8
import os
import re
import time
import pickle
import shutil
import paramiko
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from scipy.spatial.distance import cityblock
from tqdm import tqdm


classes = ['car',        # 1
           'car_fr',     # 2
           'bicycle',    # 3
           'person',     # 4
           'cyclist',    # 5
           'tricycle']   # 6


def get_normed_hist(frame, hist_size=64):
    '''
    计算直方图并归一化、扁平化(展开3个通道)
    @return 1维数组, 元素个数hist_size*通道数
    '''
    color_hist = [cv2.calcHist([frame], [c], None, [hist_size], [0.0, 255.0])
                  for c in range(3)]
    color_hist = np.array([hist_c / float(sum(hist_c))
                           for hist_c in color_hist])
    return color_hist.flatten()  # 将3个通道展开成1维


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# -----------------------------------图像批量处理


def png2jpg(src_dir, dst_dir):
    """
    batch processing to convert png files to jpg files
    """
    if not (os.path.isdir(src_dir) and os.path.isdir(dst_dir)):
        print('=> [Err]: invalid src or dst dir.')
        return

    for f_name in os.listdir(src_dir):
        if f_name.endswith('.png'):
            f_path = src_dir + '/' + f_name
            if os.path.isfile(f_path):
                img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                dst_path = dst_dir + '/' + f_name.replace('.png', '.jpg')
                if not os.path.isfile(dst_path):
                    cv2.imwrite(dst_path, img)
                    print('=> %s converted to jpg file.' % f_name)


def resize_jpgs(src_dir, width, height):
    """
    batch processing to resize jpg images
    """
    if not os.path.isdir(src_dir):
        print('=> invalid src dir.')
        return

    for f_name in os.listdir(src_dir):
        if f_name.endswith('.jpg'):
            f_path = src_dir + '/' + f_name
            if os.path.isfile(f_path):
                img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                img_cvt = cv2.resize(img, (height, width),
                                     interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f_path, img_cvt)
                print('=> %s resized to (%d, %d)' % (f_name, width, height))


def videos2keyframe(videos_list,
                    img_root,
                    threshold=0.001):  # 0.12
    """
    处理一组视频
    @TODO: 多线程加速
    """
    # for x in videos_list:
    #     assert os.path.exists(x)

    if not os.path.exists(img_root):
        os.makedirs(img_root)
    # for x in os.listdir(img_root):
    #     if x.endswith('.jpg'):
    #         x_path = os.path.join(img_root, x)
    #         os.remove(x_path)

    existed_imgs = [x for x in os.listdir(img_root) if x.endswith('.jpg')]
    existed_imgs.sort(key=lambda x: int(re.match('det.*', x).group(1)))
    count = len([x for x in os.listdir(img_root) if x.endswith('.jpg')])
    print('=> initial count: ', count)

    for x in tqdm(videos_list):
        # 每个视频单独计数
        count = 0

        seq_name = os.path.split(x)[-1]
        img_dir = img_root + '/' + seq_name.split('.')[0] + '_imgs'
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        else:
            shutil.rmtree(img_dir)
            os.makedirs(img_dir)

        batch_id = re.match('.*(\d+)\.mp4', x).group(1)
        cap = cv2.VideoCapture(x)

        print('--processing video %s' % x)
        FRAME_NUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频所有帧数
        print('Total {:d} frames'.format(FRAME_NUM))

        if FRAME_NUM == 0:
            break

        for i in range(1, FRAME_NUM):
            success, frame = cap.read()
            if not success:  # 判断当前帧是否存在
                break

            #   提取关键帧
            if i == 1:  # 记录往前2帧
                frame_prev_2 = frame
                continue
            elif i == 2:  # 记录往前1帧
                frame_prev_1 = frame
                continue
            else:  # 从第3帧开始判断
                hist_prev_1 = get_normed_hist(frame_prev_1)
                hist_prev_2 = get_normed_hist(frame_prev_2)
                hist_cur = get_normed_hist(frame)

                # 计算两个histgram的曼哈顿距离
                score_pre = cityblock(hist_prev_1, hist_prev_2)
                score_cur = cityblock(hist_cur, hist_prev_1)

                # 一阶, 二阶差分阈值判断
                if (score_cur >= threshold) and (abs(score_cur - score_pre) >= threshold * 0.0001):

                    # 模糊检测过滤
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fm = variance_of_laplacian(gray)
                    if fm and fm < 1000.0:  # 535.0
                        print('Skip frame {:d}'.format(i))
                        continue

                    date_name = time.strftime(
                        '_%Y_%m_%d_', time.localtime(time.time()))

                    write_name = img_dir + '/' + '{:05d}.jpg'.format(count)
                    # cv2.putText(frame, '%d' % fm, (80, 80), 0, 2, (0,
                    # 255, 255), 2)
                    print('=> write key frame %s' % write_name)
                    cv2.imwrite(write_name, frame)
                    count += 1

                # 更新前两帧
                frame_prev_2 = frame_prev_1  # 更新往前1帧
                frame_prev_1 = frame  # 更新往前2帧

    cap.release()  # 释放资源
    print('=> total {} imgs'.format(count))


def split_to_JPG_Anno(path):
    """
    :param path:
    :return:
    @TODO: 多线程异步加速
    """
    # parent_dir = os.path.split(os.path.realpath(path))[0]
    # print('=> parent_dir: ', parent_dir)
    JPEG_dir = os.path.join(path, 'JPEGImages')
    if not os.path.exists(JPEG_dir):
        os.makedirs(JPEG_dir)
    Annos_dir = os.path.join(path, 'Annotations')
    if not os.path.exists(Annos_dir):
        os.makedirs(Annos_dir)

    # train_txt = os.path.join(path, 'train.txt') # file list txt
    # f = open(train_txt, 'w')

    for x in tqdm(os.listdir(path)):
        if x.endswith('.jpg'):
            jpg_path = os.path.join(path, x)
            shutil.copy(jpg_path, JPEG_dir)

            # jpg_file_path = os.path.join(JPEG_dir, x)
            # f.write(jpg_file_path + '\n')
        elif x.endswith('.xml'):
            xml_path = os.path.join(path, x)
            shutil.copy(xml_path, Annos_dir)
    # f.close()


if __name__ == '__main__':
    # --------------------------------------
    # video_list = ['f:mcmot_seq1/car_8.mp4',
    #    'f:/car_9.mp4']
    video_list = ['f:/mcmot_seq4.mp4']
    videos2keyframe(videos_list=video_list,
                    img_root='f:/seq_img_root',
                    threshold=0.01)

    #png2jpg('F:/2019420101148_even', 'F:/2019420101148_even')
    #resize_jpgs('F:/2019420101148_even', 200, 150)

    # split_to_JPG_Anno('f:/car_19_0118')
