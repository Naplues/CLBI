# -*- coding:utf-8 -*-

import warnings
from utils.helper import *
from sklearn import metrics
from utils.eval import evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

root_path = r'C://Users/GZQ/Desktop/CLDP_data'


# 版本内预测实验设计
def within_release_prediction(proj, num_iter=10, num_folds=10, model=None, th=50, path=''):
    """
    版本内预测
    :param proj: 项目版本名
    :param num_iter: 重复次数 默认10
    :param num_folds:折数 默认 10
    :param model
    :param th
    :param path
    :return:
    """
    make_path(path + proj + '/')
    log = '=' * 10 + ' Within-release prediction for ' + proj + ' ' + '=' * 60
    print(log[:60])
    # 声明储存预测结果变量
    oracle_list = []
    prediction_list = []
    mcc_list = []

    # Line-level指标
    performance = 'Setting,Test release,Recall,FAR,d2h,MCC,Recall@20%,IFA_mean,IFA_median\n'
    # 读取数据
    text, text_lines, labels, filenames = read_file_level_dataset(proj)
    # 重复10次实验
    for it in range(num_iter):
        print('=' * 20 + ' Running iter ' + str(it) + ' ' + '=' * 20)
        # 定义10-折划分设置
        ss = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=it)
        fold = 0
        for train_index, test_index in ss.split(text, labels):
            print('=' * 10 + ' Running fold ' + str(fold) + ' ' + '=' * 10)
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
            classifier = LogisticRegression().fit(train_vtr, train_label)
            test_predictions = classifier.predict(test_vtr)

            # 4. 评估并储存预测结果极其评估指标
            oracle_list.append(test_labels)
            prediction_list.append(test_predictions)
            mcc_list.append(metrics.matthews_corrcoef(test_labels, test_predictions))

            # 5. 解释代码行级别的缺陷概率
            out_file = path + proj + '/wr_' + str(it) + '_' + str(fold) + '_line_risk_ranks.pk'
            # 如果模型的结果已经存在直接进行评估, 否则重新进行预测并评估
            if os.path.exists(out_file):
                with open(out_file, 'rb') as file:
                    data = pickle.load(file)
                    # with open(root_path + '/Result/LineDP/cr_line_level_ranks_' + test_proj + '.pk',  'rb') as f:
                    # cut_data = pickle.load(f) cut_data[1], cut_data[2]
                    performance += evaluation(proj, data[0], data[1], data[2], data[3], data[4])
            else:
                performance += model(proj, vector, test_text_lines, test_filename, test_predictions, out_file, th)

            fold += 1
    # 将行级别的性能评估结果写入文件
    save_csv_result(path + proj + '/wr_line_level_evaluation_.csv', performance)
    dump_pk_result(path + proj + '/within_release.pk', [oracle_list, prediction_list, mcc_list])
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))
