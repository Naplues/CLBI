# -*- coding:utf-8 -*-

import warnings
from utils.helper import *
from sklearn import metrics
from utils.eval import evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

root_path = r'C://Users/GZQ/Desktop/CLDP_data'


# 版本间预测实验
def cross_release_prediction(proj, releases, model, th, path):
    """
    版本间预测
    :param proj: target project
    :param releases:
    :param model A prediction model
    :param th
    :param path
    :return:
    """

    log = '=' * 10 + ' Cross-release prediction for ' + proj + ' ' + '=' * 60
    print(log[:60])
    # 声明储存预测结果变量
    oracle_list = []
    prediction_list = []
    # 声明储存评估指标变量
    mcc_list = []

    # Line-level指标
    performance = 'Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median,MRR,MAP,IFA list\n'
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
        clf = LogisticRegression().fit(train_vtr, train_label)
        test_predictions = clf.predict(test_vtr)

        # 4. 储存文件级别的预测结果和评估指标
        oracle_list.append(test_label)
        prediction_list.append(test_predictions)
        mcc_list.append(metrics.matthews_corrcoef(test_label, test_predictions))

        # 5. 解释代码行级别的缺陷概率
        out_file = path + 'cr_line_level_ranks_' + test_proj + '.pk'

        # 如果模型的结果已经存在直接进行评估, 否则重新进行预测并评估
        if os.path.exists(out_file):
            with open(out_file, 'rb') as file:
                data = pickle.load(file)
                # with open(root_path + '/Result/LineDP/cr_line_level_ranks_' + test_proj + '.pk',  'rb') as f:
                # cut_data = pickle.load(f) cut_data[1], cut_data[2]
                performance += evaluation(proj, data[0], data[1], data[2], data[3], data[4])
        else:
            performance += model(test_proj, vector, clf, test_text_lines, test_filename, test_predictions, out_file, th)

    # 输出行级别的结果
    save_csv_result(path + 'cr_line_level_evaluation_' + proj + '.csv', performance)
    dump_pk_result(path + 'cr_file_level_evaluation_' + proj + '.pk', [oracle_list, prediction_list, mcc_list])
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))
