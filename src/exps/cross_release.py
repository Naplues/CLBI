# -*- coding:utf-8 -*-

import warnings
import numpy as np

from src.utils.helper import *
from sklearn import metrics
from src.utils.eval import evaluation
from src.models.explain import *

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)


def predict_cross_release(proj, releases, model, th, path):
    """
    Predict the results under cross release experiments
    :param proj: target project
    :param releases:
    :param model: A prediction model, AccessModel or LineDPModel
    :param th:
    :param path:
    :return:
    """
    print(f'========== Cross-release prediction for {proj} =================================================='[:60])
    # 声明储存预测结果变量
    oracle_list = []
    prediction_list = []
    # 声明储存评估指标变量
    mcc_list = []

    print("Training set\t ===> \tTest set.")
    for i in range(len(releases) - 1):
        # 1. 读取数据 训练版本的索引为 i, 测试版本的索引为 i + 1
        train_proj, test_proj = releases[i], releases[i + 1]
        print("%s\t ===> \t%s" % (train_proj, test_proj))
        #    源码文本列表 源码文本行级别列表 标签列表 文件名称
        train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(train_proj)
        test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(test_proj)

        # 2. 定义一个矢量器. 拟合矢量器, 将文本特征转换为数值特征
        vector = CountVectorizer(lowercase=False, min_df=2)
        train_vtr = vector.fit_transform(train_text)
        test_vtr = vector.transform(test_text)

        # 3. 定义 LogisticRegression 分类器, 使用默认设置进行训练和预测
        if model == TMI_RF_Model:
            print('Random Forest')
            clf = RandomForestClassifier().fit(train_vtr, train_label)
        elif model == TMI_Tree_Model:
            print('Decision Tree')
            clf = DecisionTreeClassifier().fit(train_vtr, train_label)
        elif model == TMI_SVM_L_Model:
            print('Linear SVM')
            clf = LinearSVC().fit(train_vtr, train_label)
        elif model == TMI_MNB_Model:
            print('MultinomialNB')
            clf = MultinomialNB().fit(train_vtr, train_label)
        else:
            print('Logistic')
            clf = LogisticRegression().fit(train_vtr, train_label)
        test_predictions = clf.predict(test_vtr)

        # 4. 储存文件级别的预测结果和评估指标
        oracle_list.append(test_label)
        prediction_list.append(test_predictions)
        mcc_list.append(metrics.matthews_corrcoef(test_label, test_predictions))

        # 5. 预测代码行级别的缺陷概率
        out_file = f'{path}cr_line_level_result_{test_proj}.pk'
        model(test_proj, vector, clf, test_text_lines, test_filename, test_predictions, out_file, th)

    dump_pk_result(f'{path}cr_file_level_result_{proj}.pk', [oracle_list, prediction_list, mcc_list])
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))


def eval_cross_release(proj, releases, path, prediction_model, depend=False):
    """
    Evaluate the results under cross release experiments
    :param proj: Target project
    :param releases: The release list in target project
    :param path: The path of result .pk file
    :param prediction_model
    :param depend: Whether depending the results of LineDP
    :return:
    """
    performance = 'Setting,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for i in range(len(releases) - 1):
        test_proj = releases[i + 1]
        print(f"Target release:\t{test_proj} ======================================================="[:80])
        out_file = f'{path}cr_line_level_result_{test_proj}.pk'
        dep_file = f'{root_path}Result/CP/{prediction_model}_50/cr_line_level_result_{test_proj}.pk'
        try:
            with open(out_file, 'rb') as file:
                data = pickle.load(file)
                if depend:
                    print('depend')
                    with open(dep_file, 'rb') as f:
                        dep_data = pickle.load(f)
                        data[3] = dep_data[3]
                performance += evaluation(test_proj, data[0], data[1], data[2], data[3], data[4])
        except IOError:
            print('Error! Not found result file %s or %s' % (out_file, dep_file))
            return

    # Output the evaluation results for line level experiments
    save_csv_result(f'{path}line_level_evaluation_{proj}.csv', performance)
