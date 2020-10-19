# coding:utf-8

import numpy as np
import os
import cv2
from tqdm import tqdm


def FindFileWithSuffix(root, suffix, f_list):
    """
    递归的方式查找特定后缀文件
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            FindFileWithSuffix(f_path, suffix, f_list)


def ChangeSuffix(root, src_suffix, dst_suffix):
    """
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root.')
        return

    f_list = []
    FindFileWithSuffix(root, src_suffix, f_list)

    for f in tqdm(f_list):
        new_f = f.replace(src_suffix, dst_suffix)
        os.rename(f, new_f)


def RMFilesWithSuffix(root, suffix):
    """
        删除指定根目录下指定后缀的所有文件
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root.')
        return

    f_list = []
    FindFileWithSuffix(root, suffix, f_list)

    for f in tqdm(f_list):
        os.remove(f)


def GenerateFileList(root, suffix, list_name):
    """
    生成指定后缀的文件名列表txt文件
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root')
        return

    f_list = []
    FindFileWithSuffix(root, suffix, f_list)

    if len(f_list) == 0:
        print('[Warning]: empty file list')
        return

    with open(root + '/' + list_name, 'w', encoding='utf-8') as f_h:
        for i, f_path in tqdm(enumerate(f_list)):
            f_name = os.path.split(f_path)[1]
            f_h.write(f_name)

            if i != len(f_list) - 1:
                f_h.write('\n')


def sample_files(file_root, interval=10):
    """
    每隔interval个文件采样一次, 违背采样到的
    """
    if not os.path.isdir(file_root):
        print('[Err]: invalid file root.')
        return

    file_paths = [file_root + '/' + x for x in os.listdir(file_root)]
    cnt = 0
    for i, file_path in enumerate(file_paths):
        if i % interval == 0:
            print('{:s} sampled.'.format(file_path))
            cnt += 1
        else:
            os.remove(file_path)
            print('{:s} removed.'.format(file_path))


def rename_files(file_root, width=2):
    """
    以编号重命名目录下所有文件
    """
    if not os.path.isdir(file_root):
        print('[Err]: invalid file root.')
        return

    file_paths = [file_root + '/' + x for x in os.listdir(file_root)]
    for i, file_path in enumerate(file_paths):
        file_name = os.path.split(file_path)[-1]
        name, suffix = file_name.split('.')
        format = '{:0' + str(width) + 'd}'
        new_name = format.format(i) + '.' + suffix
        rename_f_path = file_root + '/' + new_name
        os.rename(file_path, rename_f_path)
        print('{:s} renamed to {:s}.'.format(file_path, rename_f_path))


def rename_files_from_txt(file_root, txt_f_path):
    """
    通过读取txt文件名称列表, 按照读取顺序重命名
    """
    if not os.path.isdir(file_root):
        print('[Err]: invalid file root.')
        return

    if not os.path.isfile(txt_f_path):
        print('[Err]: invalid txt file.')
        return

    with open(txt_f_path, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines()]
        print('Total {:d} files.'.format(len(lines)))

        for i, line in enumerate(lines):
            old_path = file_root + '/' + line
            if not os.path.isfile(old_path):
                print('[Warning]: {:s} not exists.')
                continue

            name, suffix = line.split('.')
            new_name = '{:04d}'.format(i) + '.' + suffix
            new_path = file_root + '/' + new_name
            os.rename(old_path, new_path)
            print('{:s} renamed to {:s}.'.format(old_path, new_path))


def ResizeImg(img_path, size=None):
    """
    """
    if not os.path.isfile(img_path):
        print("[Err]: invalid path")
        return

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("[Err]: empty image")
        return

    new_size = size
    if new_size is None:
        new_size = (int(img.shape[1] * 0.24 + 0.5),
                    int(img.shape[0] * 0.24 + 0.5))
        dst = cv2.resize(img,
                         new_size,
                         cv2.INTER_NEAREST)
    else:
        dst = cv2.resize(img, new_size, cv2.INTER_NEAREST)

    cv2.imwrite(img_path, dst)
    print('{:s} resized to [{:d}, {:d}]'.format(
        img_path, new_size[0], new_size[1]))


def ResizeImages(img_root, size=None):
    img_paths = [img_root + '/' +
                 x for x in os.listdir(img_root) if x.endswith('.jpg')]
    print('Total {:d} images.'.format(len(img_paths)))

    for img_path in img_paths:
        ResizeImg(img_path, size)


if __name__ == '__main__':
    # ChangeSuffix(root='f:/workspace/resultPro/depth_maps',
    #              src_suffix='bin.jpg',
    #              dst_suffix='_r14_th0.jpg')

    # RMFilesWithSuffix(root='d:/workspace/resultPro/depth_maps',
    # 				  suffix='_sigma_alpha_0_2_.jpg')

    # GenerateFileList(root='f:/ETH3D/multi_view_training_dslr_undistorted/terrains',
    # 				 suffix='.JPG',
    # 				 list_name='bundler.out.list.txt')

    # sample_files(file_root='f:/tmp', interval=10)

    # rename_files(file_root='e:/CppProjs/OpencvSFM/dog')

    rename_files_from_txt(file_root='e:/CppProjs/OpencvSFM/dog',
                          txt_f_path='e:/CppProjs/OpencvSFM/dog/bundler.out.list.txt')
