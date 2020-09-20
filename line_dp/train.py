# -*- coding:utf-8-*-
from warnings import simplefilter

import os
import re
import pickle
import numpy as np

from pprint import pprint
from sklearn import metrics
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
simplefilter(action='ignore', category=FutureWarning)

# 全局变量设置
root_path = r'C://Users/GZQ/Desktop/CLDP_data'
file_level_path = root_path + '/Dataset/File-level/'
result_path = root_path + '/Result/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


# 读取项目列表
def get_project_list(folder):
    return [file.replace(file_level_path_suffix, '') for file in os.listdir(folder)]


# 读取文件级别的数据集信息
def read_file_level_dataset(proj):
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
        texts = []
        for t in texts_lines:
            texts.append(' '.join(t))

        return texts, texts_lines, numeric_labels, src_files


# 读取代码行级别的数据集信息
def read_line_level_dataset(proj):
    path = file_level_path + proj + line_level_path_suffix
    with open(path, 'r', encoding='utf-8', errors='ignore')as file:
        lines = file.readlines()
        file_buggy_lines = {}
        for line in lines:
            temp = line.split(',')
            if temp[0] not in file_buggy_lines:
                file_buggy_lines[temp[0]] = temp[1]
            else:
                file_buggy_lines[temp[0]].append(temp[1])

    return file_buggy_lines


# 版本内预测实验
def within_release_prediction(proj, num_iter=10, num_folds=10):
    """
    版本内预测
    :param proj: 项目版本名
     :param num_iter: 重复次数 默认10
    :param num_folds:折数 默认 10
    :return:
    """
    print('Within-release prediction for ' + proj)
    # 声明储存预测结果变量
    test_list = []
    prediction_list = []
    # 声明储存评估指标变量
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []

    # 读取数据
    text, text_lines, labels, filenames = read_file_level_dataset(proj)
    # 重复10次实验
    for it in range(num_iter):
        print('Running %d-fold' % it)
        # 定义10-折划分设置
        ss = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=it)
        for train_index, test_index in ss.split(text, labels):
            # 1. 取出每折原始数据
            train_text = np.array(text)[train_index]
            train_label = np.array(labels)[train_index]

            test_text = np.array(text)[test_index]
            test_text_lines = np.array(text_lines)[test_index]
            test_labels = np.array(labels)[test_index]

            # 2. 定义一个矢量器
            vector = CountVectorizer(lowercase=False, min_df=2)
            # 拟合矢量器, 将文本特征转换为数值特征
            train_vtr = vector.fit_transform(train_text)
            test_vtr = vector.transform(test_text)

            # 3. 定义 LogisticRegression 分类器进行预测
            classifier = LogisticRegression(max_iter=5000).fit(train_vtr, train_label)
            test_predictions = classifier.predict(test_vtr)

            # 4. 评估并储存预测结果极其评估指标
            test_list.append(test_labels)
            prediction_list.append(test_predictions)
            precision_list.append(metrics.precision_score(test_labels, test_predictions))
            recall_list.append(metrics.recall_score(test_labels, test_predictions))
            f1_list.append(metrics.f1_score(test_labels, test_predictions))
            mcc_list.append(metrics.matthews_corrcoef(test_labels, test_predictions))

            # 5. 解释代码行级别的缺陷概率
            out_file = result_path + 'with_predictions_' + proj + str(it) + '_line_risk_ranks.pk'
            line_dp(vector, classifier, test_text, test_text_lines, test_predictions, out_file)

    # 打印平均结果
    print('\nAvg P:\t%.3f' % np.average(precision_list))
    print('Avg R:\t%.3f' % np.average(recall_list))
    print('Avg F1:\t%.3f' % np.average(f1_list))
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))
    with open(result_path + 'within_predictions_' + proj + '.pk', 'wb') as file:
        pickle.dump([test_list, prediction_list, precision_list, recall_list, f1_list, mcc_list], file)


# 版本间预测实验
def cross_release_prediction(proj, releases_list):
    """
    版本间预测
    :param proj: 目标项目
    :param releases_list:
    :return:
    """
    print('Within-release prediction for ' + proj)
    # 声明储存预测结果变量
    test_list = []
    pred_list = []
    # 声明储存评估指标变量
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []
    print("Training set,\tTest set.")
    for i in range(len(releases_list) - 1):
        print("%s,\t%s" % (releases_list[i], releases_list[i + 1]))
        # 1. 读取数据 训练数据索引为 i 测试数据索引为 i+1
        # 文本列表 行级别文本列表 标记列表 文件名称
        train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(releases_list[i])
        test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(releases_list[i + 1])

        # 2. 定义一个矢量器
        vector = CountVectorizer(lowercase=False, min_df=2)
        # 拟合矢量器, 将文本特征转换为数值特征
        train_vtr = vector.fit_transform(train_text)
        test_vtr = vector.transform(test_text)

        # 3. 定义 LogisticRegression 分类器进行预测
        classifier = LogisticRegression(max_iter=5000).fit(train_vtr, train_label)
        test_predictions = classifier.predict(test_vtr)

        # 4. 评估并储存预测结果极其评估指标
        test_list.append(test_label)
        pred_list.append(test_predictions)
        precision_list.append(metrics.precision_score(test_label, test_predictions))
        recall_list.append(metrics.recall_score(test_label, test_predictions))
        f1_list.append(metrics.f1_score(test_label, test_predictions))
        mcc_list.append(metrics.matthews_corrcoef(test_label, test_predictions))

        # 5. 解释代码行级别的缺陷概率
        out_file = result_path + 'cross_predictions_' + proj + '_line_risk_ranks.pk'
        line_dp(vector, classifier, test_text, test_text_lines, test_predictions, out_file)
        # break  # TODO 待删除

    # 打印平均结果
    print('\nAvg P:\t%.3f' % np.average(precision_list))
    print('Avg R:\t%.3f' % np.average(recall_list))
    print('Avg F1:\t%.3f' % np.average(f1_list))
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))
    with open(result_path + 'cross_predictions_' + proj + '.pk', 'wb') as file:
        pickle.dump([test_list, pred_list, precision_list, recall_list, f1_list, mcc_list], file)


# 进行代码行级别的排序
def line_dp(vector, classifier, test_text, test_text_lines, test_predictions, out_file):
    # 5. 解释代码行级别的缺陷概率
    # 预测值为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])
    # 制作管道
    c = make_pipeline(vector, classifier)
    # 定义解释器
    explainer = LimeTextExplainer(class_names=['defect', 'non-defect'])
    # 文本分词器
    tokenizer = vector.build_tokenizer()
    # hit_count_list
    hit_line_list = []
    # 待解释的文件
    for target_file_index in defect_prone_file_indices:
        # 目标文件
        target_file = test_text[target_file_index]
        # 对分类结果进行解释
        exp = explainer.explain_instance(target_file, c.predict_proba, num_features=50)
        # 取出risk tokens, 取前20个, 可能不足20个 TODO
        positive_tokens = [i[0] for i in exp.as_list() if i[1] > 0][:20]
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]
        # 20% effort 的cut-off
        cut_off = .2 * len(target_file_lines)
        # risk tokens 的命中次数, 初始为0
        hit_count = np.array([0] * len(target_file_lines))

        # 统计每行代码中出现risk tokens的个数
        for index in range(len(target_file_lines)):
            tokens_in_line = tokenizer(target_file_lines[index])
            # 检测每个risk token在代码中的出现位置
            for risk_token in positive_tokens:
                if risk_token in tokens_in_line:
                    hit_count[index] += 1

        # 根据命中次数对代码行进行降序排序, 按照排序后数值从大到小的顺序显示代码行在原列表中的索引, cut_off 为切分点
        hit_line_list.append(np.argsort(-hit_count)[:cut_off])

        print('%d of %d files ranked finish!' % (len(hit_line_list), len(defect_prone_file_indices)))

    # 计算评估指标 可能需要文件名
    for hit_line in hit_line_list:
        pass
    read_line_level_dataset()
    with open(out_file, 'wb') as file:
        pickle.dump(hit_line_list, file)


# ################# 运行版本内预测实验 ###################
def run_within_release_prediction():
    for project in get_project_list(file_level_path):
        within_release_prediction(proj=project, num_iter=10, num_folds=10)


# ################# 运行版本间预测实验 ###################
def run_cross_release_prediction():
    release_list = get_project_list(file_level_path)
    projects_dict = {}
    for release in release_list:
        project = release.split('-')[0]
        if project not in projects_dict:
            projects_dict[project] = [release.replace(file_level_path_suffix, '')]
        else:
            projects_dict[project].append(release.replace(file_level_path_suffix, ''))
    for project, releases in projects_dict.items():
        cross_release_prediction(proj=project, releases_list=releases)


if __name__ == '__main__':
    # 运行版本内预测实验
    # run_within_release_prediction()
    # 运行版本间预测实验
    run_cross_release_prediction()
