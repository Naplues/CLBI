# -*- coding:utf-8 -*-

from warnings import simplefilter

import os
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
simplefilter(action='ignore', category=FutureWarning)

# 全局变量设置
root_path = r'D:/CLDP_data/'  # r'C://Users/gzq/Desktop/CLDP_data/'  r'D:/CLDP_data/'
file_level_path = f'{root_path}Dataset/File-level/'
line_level_path = f'{root_path}Dataset/Line-level/'
result_path = f'{root_path}Result/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'

projects = [
    'ambari',
    'amq',
    'bookkeeper',
    'calcite',
    'camel',
    'cassandra',
    'flink',
    'groovy',
    'hbase',
    'hive',
    'ignite',
    'log4j2',
    'mahout',
    'mng',
    'nifi',
    'nutch',
    'storm',
    'tika',
    'ww',
    'zookeeper',

    # 'activemq',
    # 'camel',
    # 'derby',
    # 'groovy',
    # 'hbase',
    # 'hive',
    # 'jruby',
    # 'lucene',
    # 'wicket',

]


def get_project_release_list():
    """
    返回项目名-版本号列表 e.g., activemq-5.0.0
    :return:
    """
    # 按照时间排好顺序的releases
    return [
        'ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0', 'ambari-2.7.0',
        'amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0',
        'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0',
        'bookkeeper-4.0.0', 'bookkeeper-4.2.0', 'bookkeeper-4.4.0',
        'calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0', 'calcite-1.15.0', 'calcite-1.16.0',
        'calcite-1.17.0', 'calcite-1.18.0',
        'camel-2.11.0', 'camel-2.12.0', 'camel-2.13.0', 'camel-2.14.0', 'camel-2.17.0', 'camel-2.18.0', 'camel-2.19.0',
        'cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11',
        'cassandra-1.2.11',
        'flink-1.4.0', 'flink-1.6.0',
        'groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1', 'groovy-1.8.7',
        'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0', 'groovy-2.5.5',
        'hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11',
        'hive-0.14.0', 'hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0',
        'ignite-1.0.0', 'ignite-1.4.0', 'ignite-1.6.0',
        'log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6', 'log4j2-2.7',
        'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10',
        'mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8',
        'mng-3.0.0', 'mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0',
        'nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0',
        'nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.9',
        'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14', 'nutch-1.15',
        'storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5',
        'tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7', 'tika-1.10',
        'tika-1.13', 'tika-1.15', 'tika-1.17',
        'ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1',
        'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20', 'ww-2.3.24',
        'zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3',
        #
        # 'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
        # 'camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0',
        # 'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1',
        # 'groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2',
        # 'hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2',
        # 'hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0',
        # 'jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0',
        # 'lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1',
        # 'wicket-1.3.0-beta2', 'wicket-1.3.0-incubating-beta-1', 'wicket-1.5.3',
    ]
    # return [file.replace(file_level_path_suffix, '') for file in os.listdir(folder)]


def get_project_releases_dict():
    """
    get project releases dict: dict[project] = [releases]
    返回 项目名 -> 版本号 字典
    :return:
    """
    release_list = get_project_release_list()

    project_releases_dict = {}
    for release in release_list:
        project = release.split('-')[0]
        if project not in project_releases_dict:
            project_releases_dict[project] = [release]
        else:
            project_releases_dict[project].append(release)

    return project_releases_dict


def read_file_level_dataset(proj, file_path=file_level_path):
    """
    读取文件级别的数据集信息
    :param proj:项目名
    :param file_path
    :return:
    """
    path = file_path + proj + file_level_path_suffix
    with open(path, 'r', encoding='utf-8', errors='ignore')as file:
        lines = file.readlines()
        # 文件索引列表, 每个文件名不一样才语句才没有错误 TODO
        src_file_indices = [lines.index(line) for line in lines if r',true,"' in line or r',false,"' in line]
        # 源文件路径,需要时返回 OK
        src_files = [lines[index].split(',')[0] for index in src_file_indices]
        # 缺陷标记
        string_labels = [lines[index].split(',')[1] for index in src_file_indices]
        numeric_labels = [1 if label == 'true' else 0 for label in string_labels]

        # 行级别的文本语料库
        texts_lines = []
        for i in range(len(src_file_indices) - 1):
            # 从当前文件名所在行开始到下一个文件名所在行结束
            code_lines = lines[src_file_indices[i]:src_file_indices[i + 1]]
            # 去掉首行中的文件名和标签,以及首行中的引号"
            code_lines[0] = code_lines[0].split(',')[-1][1:]
            # 删除列表中最后的"
            del code_lines[-1]
            texts_lines.append(code_lines)

        # 从当前文件名所在行开始到结束
        code_lines = lines[src_file_indices[-1]:]
        # 去掉首行中的文件名和标签, 以及首行中的引号
        code_lines[0] = code_lines[0].split(',')[-1][1:]
        # 删除列表中最后的"
        del code_lines[-1]
        texts_lines.append(code_lines)

        # 多行合并后的文本语料库
        texts = [' '.join(line) for line in texts_lines]

        return texts, texts_lines, numeric_labels, src_files


def read_line_level_dataset(proj):
    """
    读取代码行级别的数据集信息
    :param proj: 项目名
    :return: 字典：dict[文件名] = [bug行号]
    """
    path = line_level_path + proj + line_level_path_suffix
    with open(path, 'r', encoding='utf-8', errors='ignore')as file:
        lines = file.readlines()
        file_buggy_lines_dict = {}
        for line in lines:
            # 跳过首行
            if line == 'File,Line_number,SRC\n':
                continue
            temp = line.split(',')
            if temp[0] not in file_buggy_lines_dict:
                file_buggy_lines_dict[temp[0]] = [int(temp[1])]
            else:
                file_buggy_lines_dict[temp[0]].append(int(temp[1]))

    return file_buggy_lines_dict


def dump_pk_result(path, data):
    """
    dump result
    :param path:
    :param data:
    :return:
    """
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pk_result(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def read_data_from_file(path):
    with open(path, 'r', encoding='utf-8', errors="ignore") as fr:
        lines = fr.readlines()
    return lines


def save_csv_result(file_path, file_name, data):
    """
    save result
    :param file_path: The file location
    :param file_name: The file name
    :param data: The data
    :return:
    """
    make_path(file_path)
    with open(f'{file_path}{file_name}', 'w', encoding='utf-8') as file:
        file.write(data)


def save_result(file_path, data):
    """
    save result

    :param file_path: The file location
    :param file_name: The file name
    :param data: The data
    :return:
    """
    make_path(file_path)
    with open(f'{file_path}', 'w', encoding='utf-8') as file:
        file.write(data)


def combine_cross_results(path):
    """
    将行级别的评估结果组合在一个文件中
    :param path:
    :return:
    """

    text_normal = 'Mode,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    text_worst = 'Mode,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for proj in projects:
        with open(f'{path}line_level_evaluation_{proj}.csv', 'r') as file:
            count = 0
            for line in file.readlines()[1:]:
                if count % 2 == 0:
                    text_normal += line
                else:
                    text_worst += line
                count += 1

    with open(f'{path}result_normal.csv', 'w') as file:
        file.write(text_normal)
    with open(f'{path}result_worst.csv', 'w') as file:
        file.write(text_worst)


def add_value(avg_data, line):
    values = line.split(',')[2:]
    for index in range(len(values)):
        avg_data[index] += float(values[index])
    return avg_data


def combine_cross_results_for_each_project(path):
    """
    将行级别的评估结果组合在一个文件中
    :param path:
    :return:
    """
    text_normal = 'Mode,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    text_worst = 'Mode,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for proj in projects:
        avg_normal, avg_worst = [.0] * 8, [.0] * 8
        with open(f'{path}line_level_evaluation_{proj}.csv', 'r') as file:
            count = 0
            for line in file.readlines()[1:]:
                if count % 2 == 0:
                    avg_normal = add_value(avg_normal, line)
                else:
                    avg_worst = add_value(avg_worst, line)
                count += 1
        avg_normal = list(np.array(avg_normal) * 2 / count)
        avg_worst = list(np.array(avg_worst) * 2 / count)
        text_normal += f'normal,{proj},' + ','.join([str(e) for e in avg_normal]) + '\n'
        text_worst += f'worst,{proj},' + ','.join([str(e) for e in avg_worst]) + '\n'

    with open(f'{path}result_normal_single_project.csv', 'w') as file:
        file.write(text_normal)
    with open(f'{path}result_worst_single_project.csv', 'w') as file:
        file.write(text_worst)


def eval_ifa(path):
    ifa_dict = {}
    with open(f'{path}result_worst.csv', 'r') as file:
        for line in file.readlines()[1:]:
            project_name = line.split(',')[1].split('-')[0]
            if project_name not in ifa_dict:
                ifa_dict[project_name] = ','.join(line.strip().split(',')[2:])
            else:
                ifa_dict[project_name] += ',' + ','.join(line.strip().split(',')[2:])

    text = ''
    for project_name in ifa_dict:
        text += project_name + ',' + ifa_dict[project_name] + '\n'

    with open(f'{path}result_worst.csv', 'w') as file:
        file.write(text)


def average(lines):
    recall, far, d2h, mcc, ce, r_20, ifa_mean, ifa_median = .0, .0, .0, .0, .0, .0, .0, .0
    release_name = ''
    for line in lines:
        ss = line.strip().split(',')
        release_name = ss[1]
        recall += float(ss[2])
        far += float(ss[3])
        d2h += float(ss[4])
        mcc += float(ss[5])
        ce += float(ss[6])
        r_20 += float(ss[7])
        ifa_mean += float(ss[8])
        ifa_median += float(ss[9])
    recall /= len(lines)
    far /= len(lines)
    d2h /= len(lines)
    mcc /= len(lines)
    ce /= len(lines)
    r_20 /= len(lines)
    ifa_mean /= len(lines)
    ifa_median /= len(lines)
    return f'{release_name},{recall},{far},{d2h},{mcc},{ce},{r_20},{ifa_mean},{ifa_median}\n'


def make_path(path):
    """
    make path is it does not exists
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def dataset_statistics():
    """
    数据集统计信息
    :return:
    """
    print('release name, #files, #buggy files, ratio, #LOC, #buggy LOC, ratio, #tokens')
    for proj in get_project_release_list():
        texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(proj)

        file_num = len(texts)
        bug_num = len([l for l in numeric_labels if l == 1])
        file_ratio = bug_num / file_num

        loc = sum([len([line for line in text if not line == ""]) for text in texts_lines])
        bug_lines = sum([len(v) for k, v in read_line_level_dataset(proj).items()])
        line_ratio = bug_lines / loc

        from sklearn.feature_extraction.text import CountVectorizer
        # 2. 定义一个矢量器. 拟合矢量器, 将文本特征转换为数值特征
        vector = CountVectorizer()
        vector.fit_transform(texts)

        tokens = len(vector.vocabulary_)
        res = (proj, file_num, bug_num, file_ratio, loc, bug_lines, line_ratio, tokens)
        print("%s, %d, %d, %f, %d, %d, %f, %d" % res)


def output_box_data_for_metric():
    setting = 'CP'
    mode = 'worst'
    index_dict = {'recall': 2, 'far': 3, 'd2h': 4, 'mcc': 5, 'ce': 6, 'r_20%': 7, 'IFA_mean': 8, 'IFA_median': 9}
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for metric_name, metric_index in index_dict.items():
        text = ', '.join([str(e) for e in thresholds]) + '\n'
        result_data = []
        for threshold in thresholds:
            path = f'{result_path}{setting}/AccessModel_{threshold}/result_{mode}.csv'
            with open(path, 'r') as file:
                tmp = []
                for line in file.readlines()[1:]:
                    tmp.append(line.split(',')[metric_index])
            result_data.append(tmp)

        for j in range(len(result_data[0])):
            for i in range(len(thresholds)):
                text += result_data[i][j] + ','
            text += '\n'

        with open(f'{result_path}{setting}/threshold_{mode}_{metric_name}.csv', 'w') as file:
            file.write(text)
    print('Finish!')


def transform():
    for release in get_project_release_list():
        data = 'filename,#total lines,#buggy lines,label\n'
        text, text_lines, label, filename = read_file_level_dataset(release)
        file_buggy = read_line_level_dataset(release)
        for index in range(len(filename)):
            name = filename[index]
            lines = len([line for line in text_lines[index] if line.strip() != ''])
            buggy = len(file_buggy[name]) if name in file_buggy.keys() else 0
            label = 1 if buggy > 0 else 0
            data += f'{name},{lines},{buggy},{label}\n'

        # save_csv_result(root_path + 'Transform/' + release + '.csv', data)
        make_path(f'{root_path}Transform')
        save_csv_result(f'{root_path}Transform/{release}.csv', data)
        print(release, 'finish')


def make_source_file():
    """
    导出所有source文件
    :return:
    """
    for release in get_project_release_list():
        # generate all source files of a release
        release_source_path = root_path + 'Dataset/Source/' + release
        make_path(release_source_path)
        text, text_lines, label, filename = read_file_level_dataset(release)
        for index in range(len(filename)):
            print(filename[index].replace('/', '.'))
            save_csv_result(release_source_path + '/' + filename[index].replace('/', '.'), text[index])
        print(len(filename), 'in', release, 'finish')


def make_udb_file():
    for release in get_project_release_list():
        # generate .udb file
        release_source_path = root_path + 'Dataset/Source/' + release
        release_udb_path = root_path + 'Dataset/UDB/' + release
        print(release_udb_path)
        os.system(f"und create -db {release_udb_path}.udb -languages java c++ python")
        os.system(f"und -db {release_udb_path}.udb add {release_source_path}")
        os.system(f"und -db {release_udb_path} -quiet analyze")


def is_test_file(src):
    """
    Whether the target source file is a test file OK
    :param src:
    :return:
    """
    return 'src/test/' in src


def is_non_java_file(src):
    """
    Whether the target source file is not a java file OK
    :param src:
    :return:
    """
    return '.java' not in src


def remove_test_or_non_java_file_from_dataset():
    """
    移除数据集中的测试文件和非java文件 OK
    :return:
    """
    for release in get_project_release_list():
        # #### remove test file from file level dataset
        t, texts_lines, numeric_labels, src_files = read_file_level_dataset(release, root_path + 'Dataset/Origin_File/')

        new_file_dataset = 'File,Bug,SRC\n'
        for index in range(len(src_files)):
            target_file = src_files[index]
            target_text = texts_lines[index]
            if is_test_file(target_file) or is_non_java_file(target_file):
                continue
            label = 'true' if numeric_labels[index] == 1 else 'false'
            new_file_dataset += f'{target_file},{label},"'
            new_file_dataset += ''.join(target_text)
            new_file_dataset += '"\n'

        out_file = file_level_path + release + file_level_path_suffix
        save_csv_result(out_file, data=new_file_dataset)

        # #### remove test file from line level dataset
        new_line_dataset = 'File,Line_number,SRC\n'
        path = root_path + 'Dataset/Origin_Line/' + release + line_level_path_suffix
        with open(path, 'r', encoding='utf-8', errors='ignore')as file:
            lines = file.readlines()
            for line in lines[1:]:
                if is_test_file(line) or is_non_java_file(line):
                    continue
                new_line_dataset += line
        out_file = line_level_path + release + line_level_path_suffix
        save_csv_result(out_file, data=new_line_dataset)
        print(release, 'finish')


def export_all_files_in_project(path):
    """
    Export all files in a specific root path OK
    :param path:
    :return:
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = (root.replace('\\', '/') + '/' + file).replace(path, '')
            if not file_path.endswith('.java') or is_test_file(file_path):
                continue
            file_list.append(file_path)
    return file_list


if __name__ == '__main__':
    # remove_test_or_non_java_file_from_dataset()
    # output_box_data_for_metric()
    # make_source_file()
    # make_udb_file()
    dataset_statistics()
