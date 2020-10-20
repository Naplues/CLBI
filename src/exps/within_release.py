# -*- coding:utf-8 -*-

import warnings
import numpy as np
from src.utils.helper import *
from sklearn import metrics
from src.utils.eval import evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)


def predict_within_release(proj_release, num_iter=10, num_folds=10, model=None, th=50, path=''):
    """
    版本内预测
    :param proj_release: Target project release
    :param num_iter: 重复次数 默认10
    :param num_folds:折数 默认 10
    :param model
    :param th
    :param path
    :return:
    """
    make_path(f'{path}{proj_release}/')
    print(
        f'========== Within-release prediction for {proj_release} =============================================='[:60])
    # 声明储存预测结果变量
    oracle_list = []
    prediction_list = []
    mcc_list = []

    # read data of each release
    text, text_lines, labels, filenames = read_file_level_dataset(proj_release)
    # 重复10次实验
    for it in range(num_iter):
        # 定义10-折划分设置
        ss = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=it)
        fold = 0
        for train_index, test_index in ss.split(text, labels):
            print(f'========== Running iter-fold: {it}-{fold} for {proj_release} ==========')
            # 1. 取出每折原始数据
            train_text = [text[i] for i in train_index]
            train_label = [labels[i] for i in train_index]

            test_text = [text[i] for i in test_index]
            test_text_lines = [text_lines[i] for i in test_index]
            test_filename = [filenames[i] for i in test_index]
            test_labels = [labels[i] for i in test_index]

            # 2. 定义一个矢量器
            vector = CountVectorizer(lowercase=False, min_df=2)
            # 拟合矢量器, 将文本特征转换为数值特征
            train_vtr = vector.fit_transform(train_text)
            test_vtr = vector.transform(test_text)

            # 3. 定义 LogisticRegression 分类器进行预测
            clf = LogisticRegression().fit(train_vtr, train_label)
            test_predictions = clf.predict(test_vtr)

            # 4. 评估并储存预测结果极其评估指标
            oracle_list.append(test_labels)
            prediction_list.append(test_predictions)
            mcc_list.append(metrics.matthews_corrcoef(test_labels, test_predictions))

            # 5. 解释代码行级别的缺陷概率
            out_file = '%s%s/wr_%d_%d.pk' % (path, proj_release, it, fold)
            model(proj_release, vector, clf, test_text_lines, test_filename, test_predictions, out_file, th)
            fold += 1
    dump_pk_result(f'{path}{proj_release}/wr_file_level_result.pk', [oracle_list, prediction_list, mcc_list])
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))


def eval_within_release(proj_release, num_iter=10, num_folds=10, path='', depend=False):
    """
    Evaluate the results under cross release experiments
    :param proj_release:
    :param num_iter:
    :param num_folds:
    :param path:
    :param depend:
    :return:
    """
    performance = 'Setting,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for it in range(num_iter):
        for fold in range(num_folds):
            print(f'========== Running iter-fold: {it}-{fold} for {proj_release} ==========')
            out_file = '%s%s/wr_%d_%d.pk' % (path, proj_release, it, fold)
            dep_file = '%sResult/WP/LineDPModel_50/%s/wr_%d_%d.pk' % (root_path, proj_release, it, fold)
            try:
                with open(out_file, 'rb') as file:
                    data = pickle.load(file)
                    if depend:
                        print('depend')
                        with open(dep_file, 'rb') as f:
                            dep_data = pickle.load(f)
                            data[3] = dep_data[3]

                    performance += evaluation(proj_release, data[0], data[1], data[2], data[3], data[4])
            except IOError:
                print('Error! Not found result file %s or %s' % (out_file, dep_file))
                return

    # Output the evaluation results for line level experiments
    save_csv_result(f'{path}{proj_release}/line_level_evaluation_{proj_release}.csv', performance)
