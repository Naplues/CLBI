# -*- coding:utf-8 -*-
from warnings import simplefilter

import os
import re

import pickle

# 忽略警告信息
simplefilter(action='ignore', category=FutureWarning)

# 全局变量设置
root_path = r'/Users/gzq/Desktop/CLDP_data/'
file_level_path = f'{root_path}Dataset/File-level/'
line_level_path = f'{root_path}Dataset/Line-level/'
result_path = f'{root_path}Result/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'

projects = ['activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket']


def get_project_release_list():
    """
    返回项目名-版本号列表 e.g., activemq-5.0.0
    :return:
    """
    # 按照时间排好顺序的releases
    return ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
            'camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0',
            'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1',
            'groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2',
            'hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2',
            'hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0',
            'jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
            'lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1',
            'wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3'
            ]

    # return [file.replace(file_level_path_suffix, '') for file in os.listdir(folder)]


def get_project_releases_dict():
    """
    get project releases dict: dict[project] = [releases]
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


# 保存结果
def dump_pk_result(out_file, data):
    with open(out_file, 'wb') as file:
        pickle.dump(data, file)


# 保存结果
def save_csv_result(out_file, data):
    with open(out_file, 'w') as file:
        file.write(data)


def combine_cross_results(path):
    """
    将行级别的评估结果组合在一个文件中
    :param path:
    :param proj
    :return:
    """

    text_normal = 'Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    text_worst = 'Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
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


def combine_within_results(path):
    """
    将行级别的评估结果组合在一个文件中
    :param path:
    :param proj
    :return:
    """
    text_normal = 'Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    text_worst = 'Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for proj in get_project_release_list():
        normal, worst = [], []
        with open(f'{path}{proj}/line_level_evaluation_{proj}.csv', 'r') as file:
            count = 0
            for line in file.readlines()[1:]:
                if count % 2 == 0:
                    normal.append(line)
                else:
                    worst.append(line)
                count += 1
        text_normal += average(normal)
        text_worst += average(worst)

    with open(f'{path}result_normal.csv', 'w') as file:
        file.write(text_normal)
    with open(f'{path}result_worst.csv', 'w') as file:
        file.write(text_worst)


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


if __name__ == '__main__':
    output_box_data_for_metric()
