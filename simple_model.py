# -*- coding:utf-8 -*-
from warnings import simplefilter
import re
import pickle
import numpy as np

from pprint import pprint
from helper import *
from sklearn import metrics
from evaluation import evaluation

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
line_level_path = root_path + '/Dataset/Line-level/'
result_path = root_path + '/Result/Simple/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


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
            test_filename = np.array(filenames)[test_index]
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
            simple(proj, vector, test_text, test_text_lines, test_filename, test_predictions, out_file)

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
    log = '=' * 10 + ' Cross-release prediction for ' + proj + ' ' + '=' * 60
    print(log[:60])
    # 声明储存预测结果变量
    test_list = []
    pred_list = []
    # 声明储存评估指标变量
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []

    # Line-level指标
    line_level_indicators = 'Test release,Recall,FAR,d2h,MCC,Recall@20%,IFA_mean,IFA_median\n'
    print("Training set\t ===> \tTest set.")
    for i in range(len(releases_list) - 1):
        train_proj, test_proj = releases_list[i], releases_list[i + 1]
        print("%s\t ===> \t%s" % (train_proj, test_proj))
        # 1. 读取数据 训练数据索引为 i 测试数据索引为 i+1
        # 文本列表 行级别文本列表 标记列表 文件名称
        train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(train_proj)
        test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(test_proj)

        # 2. 定义一个矢量器
        vector = CountVectorizer(lowercase=False, min_df=2)
        # 拟合矢量器, 将文本特征转换为数值特征
        train_vtr = vector.fit_transform(train_text)
        test_vtr = vector.transform(test_text)

        # 3. 定义 LogisticRegression 分类器进行预测
        clf = LogisticRegression(max_iter=10000).fit(train_vtr, train_label)
        test_predictions = clf.predict(test_vtr)

        # 4. 评估并储存预测结果极其评估指标
        test_list.append(test_label)
        pred_list.append(test_predictions)
        precision_list.append(metrics.precision_score(test_label, test_predictions))
        recall_list.append(metrics.recall_score(test_label, test_predictions))
        f1_list.append(metrics.f1_score(test_label, test_predictions))
        mcc_list.append(metrics.matthews_corrcoef(test_label, test_predictions))

        # 5. 解释代码行级别的缺陷概率
        out_file = result_path + 'cr_line_level_ranks_' + test_proj + '.pk'
        r = simple(test_proj, vector, test_text, test_text_lines, test_filename, test_predictions, out_file)
        line_level_indicators += r

    # 输出行级别的结果
    with open(result_path + 'cr_line_level_evaluation_' + proj + '.csv', 'w') as file:
        file.write(line_level_indicators)

    # 打印文件级别的平均结果
    print('\nAvg P:\t%.3f' % np.average(precision_list))
    print('Avg R:\t%.3f' % np.average(recall_list))
    print('Avg F1:\t%.3f' % np.average(f1_list))
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))
    with open(result_path + 'cr_file_level_evaluation_' + proj + '.pk', 'wb') as file:
        pickle.dump([test_list, pred_list, precision_list, recall_list, f1_list, mcc_list], file)


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


# 进行代码行级别的排序
def simple(proj, vector, test_text, test_text_lines, test_filename, test_predictions, out_file):
    """
    Line-level ranking
    :param proj:
    :param vector:
    :param test_text:
    :param test_text_lines:
    :param test_filename:
    :param test_predictions:
    :param out_file:
    :return:
    """
    # 预测值为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])
    # 文本分词器
    tokenizer = vector.build_tokenizer()

    # 读取行级别的数据集,返回一个字典变量: oracle[filename] = [line numbers]
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    # 一个源文件的排序列表中前多少行有bug
    defect_cut_off_dict = {}
    effort_cut_off_dict = {}

    # 如果结果已经存在,直接读取并评估
    if os.path.exists(out_file):
        with open(out_file, 'rb') as file:
            data = pickle.load(file)
            ranked_list_dict = data[1]
            defect_cut_off_dict = data[2]
            effort_cut_off_dict = data[3]
            # 评估,oracle, predict, cut_off
            return evaluation(proj, oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict)

    # 待解释的文件
    for target_file_index in defect_prone_file_indices:
        # 目标文本
        target_text = test_text[target_file_index]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]
        # cut-off: 20% effort 的切分点,用于将排序结果转换为分类结果
        effort_cut_off_dict[target_file_name] = int(.2 * len(target_file_lines))

        # 统计 risk tokens 的命中次数, 初始为0
        # ############################ 重点,怎么给每行赋一个缺陷值 ################################
        hit_count = np.array([.0] * len(target_file_lines))
        for index in range(len(target_file_lines)):
            tokens_in_line = tokenizer(target_file_lines[index])
            # 风险值预测
            # f1 按照自然位置排序
            f1 = 1. / (index + 1)
            # f2 按照单词个数排序
            f2 = len(tokens_in_line)
            # f3 按照方法调用深度进行排序
            f3 = call_depth(target_file_lines[index])
            # f4 按照方法调用次数进行排序
            f4 = call_number(target_file_lines[index])

            hit_count[index] += f2
        # ############################ 重点,怎么给每行赋一个缺陷值 ################################

        # 根据命中次数对代码行进行降序排序, 按照排序后数值从大到小的顺序显示代码行在原列表中的索引, cut_off 为切分点
        # line + 1,因为下标是从0开始计数而不是从1开始
        ranked_list_dict[target_file_name] = [line + 1 for line in np.argsort(-hit_count).tolist()]
        defect_cut_off_dict[target_file_name] = int(len(hit_count.tolist()) / 2)
    with open(out_file, 'wb') as file:
        pickle.dump([oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict], file)

    # 评估,oracle, predict, cut_off
    return evaluation(proj, oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict)


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
