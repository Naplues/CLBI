# -*- coding: utf-8 -*-

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


def get_same_files(prefix, prev_release, next_release):
    prev_files = set(os.listdir(f'{prefix}{prev_release}/'))
    next_files = set(os.listdir(f'{prefix}{next_release}/'))
    intersection = list(prev_files.intersection(next_files))
    for i in range(len(intersection)):
        intersection[i] = intersection[i].replace(f'{prefix}{prev_release}/', '').replace('.', '/')[:-5] + '.java'
    return set(intersection)


def compare(prefix, prev_release, next_release):
    prev_path = f'{prefix}{prev_release}/'
    next_path = f'{prefix}{next_release}/'
    prev_files = set(os.listdir(prev_path))
    next_files = set(os.listdir(next_path))
    intersection = list(prev_files.intersection(next_files))
    same_files = len(intersection)
    print(f'{len(prev_files)} ∩ {len(next_files)} = {same_files}, '
          f'Specific: {len(prev_files - next_files)} in prev release, {len(next_files - prev_files)} in next release.')
    diff_file_name = f'diff/{prev_release}-{next_release}.diff'
    with open(diff_file_name, 'w') as f:
        f.truncate()
    for i in range(same_files):
        file = intersection[i]
        print(f'{i}/{same_files}', file)
        os.system(f'diff -B -q {prev_path}{file} {next_path}{file} >> {diff_file_name}')


def get_diff_files(prefix, prev_release, next_release):
    diff_file_name = f'diff/{prev_release}-{next_release}.diff'
    files = read_data_from_file(diff_file_name)
    for i in range(len(files)):
        files[i] = files[i].split(' ')[1]
        files[i] = files[i].replace(f'{prefix}{prev_release}/', '').replace('.', '/')[:-5] + '.java'
    return set(files)


def get_defective_files(release):
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    files = [src_files[i] for i in range(len(src_files)) if numeric_labels[i] == 1]
    return set(files), set(src_files)


def fun():
    text = 'prev_rel,next_rel,diff files,'
    prefix_path = f'{root_path}Dataset/Source/'
    for project, releases in get_project_releases_dict().items():
        if project != 'amq':
            continue
        for i in range(len(releases) - 1):
            prev_rel, next_rel = releases[i], releases[i + 1]
            # compare(prefix_path, prev_rel, next_rel)
            # 两个版本中的同名文件
            same_files = get_same_files(prefix_path, prev_rel, next_rel)
            # 同名文件中发生变化的文件
            diff_files = get_diff_files(prefix_path, prev_rel, next_rel)
            # 同名文件中未发生变化的文件
            no_diff_files = same_files - diff_files

            buggy_1, all_1 = get_defective_files(prev_rel)
            buggy_2, all_2 = get_defective_files(next_rel)

            # 同名的bug文件
            insert3 = buggy_1.intersection(buggy_2)
            buggy_files_with_same_name = len(insert3)
            # 未发生变动的文件
            insert3 = insert3.intersection(no_diff_files)
            buggy_files_with_same_name_same_text = len(insert3)

            ratio_of_same_text = int(round(buggy_files_with_same_name_same_text / buggy_files_with_same_name, 3) * 100)
            string_of_same_text = f'{buggy_files_with_same_name_same_text}/{buggy_files_with_same_name}'
            text += f'{prev_rel}-{next_rel},{string_of_same_text},{ratio_of_same_text}'
            print(f'{prev_rel}-{next_rel},{string_of_same_text},{ratio_of_same_text}%')
        break
    # save_result('result.csv', text)
    pass


if __name__ == '__main__':
    fun()
    pass
