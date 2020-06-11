# encoding=utf-8

import os
import re


# |Epoch  1 |lr 0.00 |train loss 1.012 |train acc 81.017% |test acc 82.839%|
# tricycle acc 79.667% |car_cover acc 95.000%
def parse_car_multi_attrib_log(log_f_path):
    """
    解析log文件找出准确率最高的epoch对应的断点文件
    按照train_acc * test_acc最大的准则判断
    """
    if not os.path.isfile(log_f_path):
        print('[Err]: invalid log file path.')
        return

    pattern = '.*Epoch[\s|\t]+([0-9]+).*train[\s]{1}acc[\s]{1}([0-9]+\.[0-9]+)%[\s]{1}\|test[\s]{1}acc ([0-9]+\.[0-9]+).*'

    max_acc = 0.0
    max_train_acc = 0.0
    max_test_acc = 0.0

    best_epoch = 0
    max_train_acc_epoch = 0
    max_test_acc_epoch = 0
    the_line_i = 0
    with open(log_f_path, 'r', encoding='utf-8') as r_h:
        lines = r_h.readlines()
        for line_i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match is None:
                pre_line = line
                continue

            match = match.groups()
            epoch = int(match[0])
            train_acc, test_acc = float(match[1]) * 0.01, float(match[2]) * 0.01

            if train_acc > max_train_acc:
                max_train_acc = train_acc
                max_train_acc_epoch = epoch
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                max_test_acc_epoch = epoch

            acc = train_acc * test_acc
            print('Line {:d} Epoch {:d} acc {:.5f}'.format(line_i + 1, epoch, acc))

            if acc > max_acc:
                max_acc = acc
                best_epoch = epoch
                the_line_i = line_i

        print(lines[the_line_i - 2], lines[the_line_i - 1])

    print('Max test acc {:.5f} @epoch {:d}'.format(max_test_acc, max_test_acc_epoch))
    print('Best epoch: {:d}, acc {:.5f}'.format(best_epoch, max_acc))


if __name__ == '__main__':
    parse_car_multi_attrib_log(log_f_path='e:/0610.log')
    print('Work done.')
