# -*- coding:utf-8 -*-
from warnings import simplefilter

import os
import re

import pickle
import numpy as np

# 忽略警告信息
simplefilter(action='ignore', category=FutureWarning)

# 全局变量设置
root_path = r'C://Users/GZQ/Desktop/CLDP_data/'
file_level_path = root_path + 'Dataset/File-level/'
line_level_path = root_path + 'Dataset/Line-level/'
result_path = root_path + 'Result/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


def get_project_list():
    """
    返回项目名-版本号列表 e.g., activemq-5.0.0
    :return:
    """
    # 按照时间排好顺序的releases
    return [  # 'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
        # 'camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0',
        # 'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1',
        # 'groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2',
        # 'hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2',
        # 'hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0',
        'jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
        'lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1',
        'wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']

    # return [file.replace(file_level_path_suffix, '') for file in os.listdir(folder)]


def get_project_releases_dict():
    """
    get project releases dict: dict[project] = [releases]
    :return:
    """
    release_list = get_project_list()

    project_releases_dict = {}
    for release in release_list:
        project = release.split('-')[0]
        if project not in project_releases_dict:
            project_releases_dict[project] = [release]
        else:
            project_releases_dict[project].append(release)

    return project_releases_dict


# 读取文件级别的数据集信息
def read_file_level_dataset(proj):
    """
    读取文件级别的数据集信息
    :param proj:项目名
    :return:
    """
    path = file_level_path + proj + file_level_path_suffix
    with open(path, 'r', encoding='utf-8', errors='ignore')as file:
        lines = file.readlines()
        # 文件索引列表, 每个文件名不一样才语句才没有错误 TODO
        src_file_indices = [lines.index(line) for line in lines if re.search(r'.java,(true|false),', line)]
        # 源文件路径,需要时返回 OK
        src_files = [lines[index].split(',')[0] for index in src_file_indices]
        # 缺陷标记
        string_labels = [lines[index].split(',')[1] for index in src_file_indices]
        numeric_labels = np.array([1 if label == 'true' else 0 for label in string_labels])

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


# 读取代码行级别的数据集信息
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


# 保存结果
def dump_pk_result(out_file, data):
    with open(out_file, 'wb') as file:
        pickle.dump(data, file)


# 保存结果
def save_csv_result(out_file, data):
    with open(out_file, 'w') as file:
        file.write(data)


# 将行级别的评估结果组合在一个文件中
def combine_results(path):
    """
    将行级别的评估结果组合在一个文件中
    :param path:
    :return:
    """
    projects = ['activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket']

    text_normal = 'Setting,Test release,Recall,FAR,d2h,MCC,Recall@20%,IFA_mean,IFA_median\n'
    text_worst = 'Setting,Test release,Recall,FAR,d2h,MCC,Recall@20%,IFA_mean,IFA_median\n'
    for proj in projects:
        with open(path + 'cr_line_level_evaluation_' + proj + '.csv', 'r') as file:
            count = 0
            for line in file.readlines()[1:]:
                if count % 2 == 0:
                    text_normal += line
                else:
                    text_worst += line
                count += 1

        with open(path + 'result_normal.csv', 'w') as file:
            file.write(text_normal)
        with open(path + 'result_worst.csv', 'w') as file:
            file.write(text_worst)
    print('Finish!')


def parse_results(proj, path):
    pass


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 数据集统计信息
def dataset_statistics():
    print('release name, #files, #buggy files, ratio, #LOC, #buggy LOC, ratio, #tokens')
    for proj in get_project_list(file_level_path):
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


if __name__ == '__main__':
    print(call_depth(''))
