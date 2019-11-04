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


if __name__ == '__main__':
    # ChangeSuffix(root='f:/workspace/resultPro/depth_maps',
    #              src_suffix='bin.jpg',
    #              dst_suffix='_r14_th0.jpg')

    # RMFilesWithSuffix(root='d:/workspace/resultPro/depth_maps',
    # 				  suffix='_sigma_alpha_0_2_.jpg')

	GenerateFileList(root='f:/ETH3D/multi_view_training_dslr_undistorted/terrains',
					 suffix='.JPG',
					 list_name='bundler.out.list.txt')

	print('--Test done.\n')
	
