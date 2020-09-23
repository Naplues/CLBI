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


# 读取项目列表,去掉后缀
def get_project_list(folder):
    return [file.replace(file_level_path_suffix, '') for file in os.listdir(folder)]


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
def dump_result(out_file, data):
    with open(out_file, 'wb') as file:
        pickle.dump(data, file)


# 将行级别的评估结果组合在一个文件中
def combine_results(proj_list, path):
    """
    将行级别的评估结果组合在一个文件中
    :param proj_list:
    :param path:
    :return:
    """
    text = 'Test release,Recall,FAR,d2h,MCC,Recall@20%,IFA_mean,IFA_median\n'
    for proj in proj_list:
        with open(path + 'cr_line_level_evaluation_' + proj + '.csv', 'r') as file:
            for line in file.readlines()[1:]:
                text += line

        with open(path + 'result.csv', 'w') as file:
            file.write(text)
    print('Finish!')


def parse_results(proj, path):
    pass


# 嵌套深度相加
def call_depth(statement):
    statement = statement.strip('\"')
    score = 0
    depth = 0
    for char in statement:
        if char == '(':
            depth += 1
            score += depth
        elif char == ')':
            depth -= 1
    return score


def call_number(statement):
    statement = statement.strip('\"')
    score = 0
    for char in statement:
        if char == '(':
            score += 1
    return score


if __name__ == '__main__':
    projects = ['activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket']
    combine_results(projects, result_path + 'LineDP_no_adjust/')

