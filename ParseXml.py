# coding: utf-8

import os
import shutil
import cv2
import xml.etree.ElementTree as ET


def parse_ign_reg_and_apply2img(img_dir, xml_f_path):
    """
    解析xml中的第一个ignore zone
    应用到目标目录下的所有文件
    """
    if not os.path.isdir(img_dir):
        print('[Err]: invalid image dir.')
        return

    if not os.path.isfile(xml_f_path):
        print('[Err]: invalid xml file.')
        return

    # ----- 读取并解析xml
    bboxes = []
    tree = ET.parse(xml_f_path)
    root = tree.getroot()
    node = root.find('markNode')
    for obj in root.iter('object'):
        target_type = obj.find('targettype')
        type_txt = target_type.text
        if 'non_interest_zone' == type_txt:
            xml_box = obj.find('bndbox')

            x_min = int(xml_box.find('xmin').text)
            x_max = int(xml_box.find('xmax').text)
            y_min = int(xml_box.find('ymin').text)
            y_max = int(xml_box.find('ymax').text)

            #                0      1      2      3
            bboxes.append([x_min, x_max, y_min, y_max])

    # 读取img目录的每一张图片, 对每一张图应用ignore zone
    for x in os.listdir(img_dir):
        x_path = img_dir + '/' + x
        img = cv2.imread(x_path)

        # 绘制每一个ignore zone
        for bbox in bboxes:
            img[bbox[2]: bbox[3], bbox[0]: bbox[1], :] = 0

        cv2.imwrite(x_path, img)
        print('{} non_interest_zone drawed.'.format(x_path))


if __name__ == '__main__':
    parse_ign_reg_and_apply2img(img_dir='f:/seq_img_root/mcmot_seq1_imgs',
                                xml_f_path='f:/seq_img_root/Annotations/track_2020_05_28_1_0.xml')
    print('Done\n')
