# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
import os
from src.utils.helper import *


def f1():
    path1 = r'C:/Users/gzq-712/Desktop/CLDP_data/Dataset/Line-level/activemq-5.0.0_defective_lines_dataset.csv'
    path2 = r'C:/Users/gzq-712/Desktop/CLDP_data/Dataset/Line-level/activemq-5.1.0_defective_lines_dataset.csv'

    path1 = r'D:/CLDP_data/Dataset/Line-level/amq-5.0.0_defective_lines_dataset.csv'
    path2 = r'D:/CLDP_data/Dataset/Line-level/amq-5.4.0_defective_lines_dataset.csv'
    data1 = read_data_from_file(path1)[1:]
    data2 = read_data_from_file(path2)[1:]

    file_list_1 = set([x.split(',')[0] + x.split(',')[1] for x in data1])
    file_list_2 = set([x.split(',')[0] + x.split(',')[1] for x in data2])

    print(len(file_list_1), len(file_list_2))

    c = file_list_1.intersection(file_list_2)
    for file in file_list_1:
        if file in file_list_2:
            # print(file)
            pass
    print(len(c))


def is_test_file(src_file):
    # return 'test/' in src_file or 'tests/' in src_file or src_file.endswith('Test.java')
    return 'src/test/' in src_file


def calc_test(data):
    test_count = 0
    for file in data:
        if not is_test_file(file):
            test_count += 1
    return test_count


def compare(prefix, prev_release, next_release):
    prev_path = f'{prefix}{prev_release}/'
    next_path = f'{prefix}{next_release}/'
    prev_files = set(os.listdir(prev_path))
    next_files = set(os.listdir(next_path))
    intersection = list(prev_files.intersection(next_files))
    print(len(prev_files), len(next_files))
    total = len(intersection)
    print(total)
    diff_file_name = f'diff/{prev_release}-{next_release}.diff'
    with open(diff_file_name, 'w') as file:
        file.truncate()
    for i in range(total):
        file = intersection[i]
        print(f'{i}/{total}', file)
        os.system(f'diff -B -q {prev_path}{file} {next_path}{file} >> {diff_file_name}')


def get_diff_files(prefix, prev_release, next_release):
    diff_file_name = f'diff/{prev_release}-{next_release}.diff'
    files = read_data_from_file(diff_file_name)
    for i in range(len(files)):
        files[i] = files[i].split(' ')[1]
        files[i] = files[i].replace(f'{prefix}{prev_release}/', '').replace('.java', '').replace('.', '/') + '.java'
    return set(files)


def get_defective_files(release):
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    files = [src_files[i] for i in range(len(src_files)) if numeric_labels[i] == 1]
    return set(files), set(src_files)


if __name__ == '__main__':
    text = ''
    prefix_path = f'{root_path}Dataset/Source/'
    for project, releases in get_project_releases_dict().items():
        for i in range(len(releases) - 1):
            prev_rel, next_rel = releases[i], releases[i + 1]
            # compare(prefix_path, prev_rel, next_rel)
            diff_files = get_diff_files(prefix_path, prev_rel, next_rel)
            buggy_1, all_1 = get_defective_files(prev_rel)
            buggy_2, all_2 = get_defective_files(next_rel)

            insert1 = diff_files.intersection(buggy_1)
            insert2 = diff_files.intersection(buggy_2)
            insert3 = buggy_1.intersection(buggy_2)

            text += f'{prev_rel}-{next_rel},{len(diff_files)},{len(insert3)},' \
                    f'{len(insert1)}/{len(buggy_1)}/{len(all_1)},{len(insert2)}/{len(buggy_2)}/{len(all_2)}\n'
            print(f'{prev_rel}-{next_rel},{len(diff_files)},{len(insert3)},'
                  f'{len(insert1)}/{len(buggy_1)}/{len(all_1)},{len(insert2)}/{len(buggy_2)}/{len(all_2)}\n')


    pass
    # save_result('result.csv', text)
