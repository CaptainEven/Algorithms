# encoding: utf-8

import os
import shutil
import re
import cv2
from collections import defaultdict


# 目标检测类别名称和id
classes = [
    'background',    # 0
    'car',           # 1
    'bicycle',       # 2
    'person',        # 3
    'cyclist',       # 4
    'tricycle'       # 5
]  # 暂时先分为6类(包括背景)

cls2id = {
    'background': 0,
    'car': 1,
    'bicycle': 2,
    'person': 3,
    'cyclist': 4,
    'tricycle': 5
}

id2cls = {
    0: 'background',
    1: 'car',
    2: 'bicycle',
    3: 'person',
    4: 'cyclist',
    5: 'tricycle'
}

# 图片数据的宽高
W, H = 1920, 1080


def dark_label2mcmot_label(data_root, viz_root=None):
    """
    将DarkLabel的标注格式: frame# n_obj [id, x1, y1, x2, y2, label]
    转化为MCMOT的输入格式:
    1. 每张图对应一个txt的label文件
    2. 每行代表一个检测目标: cls_id, track_id, center_x, center_y, bbox_w, bbox_h(每个目标6列)
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root')
        return

    img_root = data_root + '/images'
    if not os.path.isdir(img_root):
        print('[Err]: invalid image root')

    # 创建标签文件根目录
    label_root = data_root + '/labels_with_ids'
    if not os.path.isdir(label_root):
        os.makedirs(label_root)
    else:
        shutil.rmtree(label_root)
        os.makedirs(label_root)

    # 为视频seq的每个检测类别设置[起始]track id
    start_id_dict = defaultdict(int)  # str => int
    for class_type in classes:
        start_id_dict[class_type] = 0

    # 记录每一个视频seq各类最大的track id
    seq_max_id_dict = defaultdict(int)

    # 遍历每一段视频seq
    for seq_name in os.listdir(img_root):
        seq_dir = img_root + '/' + seq_name
        print('\nProcessing seq', seq_dir)

        # 为该视频seq创建label目录
        seq_label_dir = label_root + '/' + seq_name
        if not os.path.isdir(seq_label_dir):
            os.makedirs(seq_label_dir)
        else:
            shutil.rmtree(seq_label_dir)
            os.makedirs(seq_label_dir)

        dark_txt_path = seq_dir + '/' + seq_name + '_gt.txt'
        if not os.path.isfile(dark_txt_path):
            print('[Warning]: invalid dark label file.')
            continue
        
        # ----- 开始一个视频seq的处理
        # 每遇到一个待处理的视频seq, 重置各类max_id为0
        for class_type in classes:
            seq_max_id_dict[class_type] = 0

        # 读取dark label(读取该视频seq的标注文件, 一行代表一帧)
        with open(dark_txt_path, 'r', encoding='utf-8') as r_h:
            # 去读视频标注文件的每一行: 每一行即一帧
            for line in r_h.readlines():
                line = line.split(',')
                f_id = int(line[0])
                n_objs = int(line[1])
                # print('\nFrame {:d} in seq {}, total {:d} objects'.format(f_id + 1, seq_name, n_objs))
                
                # 存储该帧所有的检测目标label信息
                fr_label_objs = []

                # 遍历该帧的每一个object
                for cur in range(2, len(line), 6):  # cursor
                    class_type = line[cur + 5].strip()
                    class_id = cls2id[class_type]  # class type => class id

                    # 解析track id
                    track_id = int(line[cur]) + 1  # track_id从1开始统计

                    # 更新该视频seq各类检测目标(背景一直为0)的max track id
                    if track_id > seq_max_id_dict[class_type]:
                        seq_max_id_dict[class_type] = track_id

                    # 根据起始track id更新在整个数据集中的实际track id
                    track_id += start_id_dict[class_type]

                    x1, y1 = int(line[cur + 1]), int(line[cur + 2])
                    x2, y2 = int(line[cur + 3]), int(line[cur + 4])

                    # 根据图像分辨率, 裁剪bbox
                    x1 = x1 if x1 >= 0 else 0
                    x1 = x1 if x1 < W else W - 1
                    y1 = y1 if y1 >= 0 else 0
                    y1 = y1 if y1 < H else H - 1
                    x2 = x2 if x2 >= 0 else 0
                    x2 = x2 if x2 < W else W - 1
                    y2 = y2 if y2 >= 0 else 0
                    y2 = y2 if y2 < H else H - 1

                    # 计算bbox center和bbox width&height
                    bbox_center_x = 0.5 * float(x1 + x2)
                    bbox_center_y = 0.5 * float(y1 + y2)
                    bbox_width = float(x2 - x1 + 1)
                    bbox_height = float(y2 - y1 + 1)

                    # bbox center和bbox width&height归一化到[0.0, 1.0]
                    bbox_center_x /= W
                    bbox_center_y /= H
                    bbox_width /= W
                    bbox_height /= H

                    # 打印中间结果, 验证是否解析正确...
                    print(track_id, x1, y1, x2, y2, class_type)
                    # if 14 == track_id:
                    #     print('pause here')

                    # 每一帧对应的label中的每一行
                    obj_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        class_id,         # class id: 从0开始计算
                        track_id,         # track id: 从1开始计算
                        bbox_center_x,    # center_x
                        bbox_center_y,    # center_y
                        bbox_width,       # bbox_w
                        bbox_height)      # bbox_h
                    # print(obj_str, end='')
                    fr_label_objs.append(obj_str)
                
                # ----- 该帧解析结束, 输出该帧的label文件
                label_f_path = seq_label_dir + '/{:05d}.txt'.format(f_id)
                with open(label_f_path, 'w', encoding='utf-8') as w_h:
                    for obj in fr_label_objs:
                        w_h.write(obj)
                print('{} written\n'.format(label_f_path))

        # 输出该视频seq各个检测类别的max track id(从1开始)
        for k, v in seq_max_id_dict.items():
            print('seq {}'.format(seq_name) + ' ' + k + ' max track id {:d}'.format(v))

        # 处理完成一个视频seq, 更新各类别start track id
        for k, v in start_id_dict.items():
            start_id_dict[k] += seq_max_id_dict[k]

    # 输出所有视频seq各个检测类别的track id总数
    print('\n')
    for k, v in start_id_dict.items():
        print(k + ' total ' + str(v) + ' track ids')


# DarkLabel格式转换代码
def cvt_dl_format(lb_f_path):
    """
    将dark label从一种格式转换成我们认为的标准格式
    """
    if not os.path.isfile(lb_f_path):
        print('[Err]: invalid label file.')
        return

    lb_path = os.path.split(lb_f_path)
    out_f_path = lb_path[0] + '/' + lb_path[1].split('.')[0] + '_cvt.txt'
    with open(out_f_path, 'w', encoding='utf-8') as w_h:
        with open(lb_f_path, 'r', encoding='utf-8') as r_h:
            for line in r_h.readlines():
                line = line.strip().split(',')
                f_id = line[0]
                n_objs = int(line[1])

                # 遍历这一帧的检测目标
                objs = []
                for cur in range(2, len(line), 5):
                    x1 = int(line[cur + 0])  # 猜测到底是第一个点(left up)还是中心点
                    y1 = int(line[cur + 1])
                    w  = int(line[cur + 2])
                    h  = int(line[cur + 3])
                    cls_id = str(line[cur + 4])

                    # img_path = 'f:/seq_data/images/mcmot_seq2_imgs/00000.jpg'
                    # img = cv2.imread(img_path)
                    # cv2.rectangle(img, (x1, y1), (x1+w, y1+h), [0, 255, 255])
                    # cv2.imshow('Test', img)
                    # cv2.waitKey()

                    # 正则表达式匹配
                    match = re.match('([a-zA-Z]+)([0-9]+)', cls_id).groups()
                    cls_name, track_id = match[0], match[1]
                    obj = [track_id, str(x1), str(y1), str(x1+w), str(y1+h), cls_name]
                    objs.append(obj)
                # print(objs)

                assert(len(objs) == n_objs)

                line_out = f_id + ',' + str(n_objs) + ','
                for obj in objs:
                    obj_str = ','.join(obj) + ','
                    line_out = line_out + obj_str
                
                line_out = line_out[:-1]
                w_h.write(line_out + '\n')
                print('frame {:d} done'.format(int(f_id)))





if __name__ == '__main__':
    dark_label2mcmot_label(data_root='f:/seq_data', viz_root=None)
    # cvt_dl_format(lb_f_path='f:/seq_data/images/mcmot_seq2_imgs_gt.txt')

    print('\nDone.')
