# -*- coding:utf-8 -*-

import warnings
import numpy as np

from src.models.access import *
from src.models.natural import LM_2_Model
from src.models.static_analysis_tools import *
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


def predict_cross_release(proj_name, releases, studied_model, depend_model=None, th=50):
    """
    Predict the results under cross release experiments: Training release i to predict test release i + 1

    :param proj_name: Target project name (i.e., Ambari)
    :param releases: Studied releases (i.e., [Ambari-2.1.0, Ambari-2.2.0, Ambari-2.4.0])
    :param studied_model: The prediction model (i.e, AccessModel)
    :param depend_model: The depend model (i.e., LineDPModel)
    :param th:
    :return:
    """
    print(f'========== Cross-release prediction for {proj_name} =============================================='[:60])
    model_name = getattr(studied_model, "__name__")
    cp_result_path = f'{root_path}Result/CP/{model_name}_{th}/'
    make_path(cp_result_path)

    # 声明储存预测结果变量
    oracle_list = []
    prediction_list = []
    # 声明储存评估指标变量
    mcc_list = []

    print("Training set\t ===> \tTest set.")
    for i in range(len(releases) - 1):
        # 1. 读取数据 训练版本的索引为 i, 测试版本的索引为 i + 1
        train_release, test_release = releases[i], releases[i + 1]
        print(f'{train_release}\t ===> \t{test_release}')

        # 源码文本列表 源码文本行级别列表 标签列表 文件名称
        train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(train_release)
        test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(test_release)

        # 2. 定义一个矢量器. 拟合矢量器, 将文本特征转换为数值特征, 定义分类器
        vector = CountVectorizer(lowercase=False, min_df=2)
        train_vtr = vector.fit_transform(train_text)
        test_vtr = vector.transform(test_text)
        clf = None

        # 如果未指定依赖模型, 依赖模型就是预测模型本身
        depend_model = studied_model if depend_model is None else depend_model

        # 3. 进行文件级别的预测, 由依赖模型决定, 得到 test_predictions
        if depend_model == AccessModel \
                or depend_model == NFC_Model or depend_model == NT_Model \
                or depend_model == PMDModel or depend_model == CheckStyleModel \
                or depend_model == LM_2_Model:
            print('File level classifier: Effort-aware ManualDown')
            file_level_out_file = f'{cp_result_path}/cr_file_level_result_{proj_name}.pk'
            if not os.path.exists(file_level_out_file):
                test_predictions = EAMD(test_text_lines)
            else:
                with open(file_level_out_file, 'rb') as file:
                    test_predictions = pickle.load(file)[1][i]
        else:
            # Define MIs-based file-level classifier
            if depend_model == TMI_LR_Model:
                print('File level classifier: Logistic')
                clf = LogisticRegression().fit(train_vtr, train_label)
            elif depend_model == TMI_MNB_Model:
                print('File level classifier: MultinomialNB')
                clf = MultinomialNB().fit(train_vtr, train_label)
            elif depend_model == TMI_SVM_Model:
                print('File level classifier: Linear SVM')
                clf = LinearSVC(random_state=0).fit(train_vtr, train_label)
            elif depend_model == TMI_DT_Model:
                print('File level classifier: Decision Tree')
                clf = DecisionTreeClassifier(random_state=0).fit(train_vtr, train_label)
            elif depend_model == TMI_RF_Model:
                print('File level classifier: Random Forest')
                clf = RandomForestClassifier(random_state=0).fit(train_vtr, train_label)
            else:
                print('File level classifier: Line-DP')
                clf = LogisticRegression().fit(train_vtr, train_label)

            test_predictions = clf.predict(test_vtr)

        # 4. 储存文件级别的预测结果和评估指标
        oracle_list.append(test_label)
        prediction_list.append(test_predictions)
        mcc_list.append(metrics.matthews_corrcoef(test_label, test_predictions))

        # 5. 预测代码行级别的缺陷概率
        print('Predicting line level defect prediction')
        out_file = f'{cp_result_path}cr_line_level_result_{test_release}.pk'
        studied_model(test_release, vector, clf, test_text_lines, test_filename, test_predictions, out_file, th)

    dump_pk_result(f'{cp_result_path}cr_file_level_result_{proj_name}.pk', [oracle_list, prediction_list, mcc_list])
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))


def eval_cross_release(proj_name, releases, studied_model, depend_model=None, th=50, depend=False):
    """
    Evaluate the results under cross release experiments

    :param proj_name: Target project name
    :param releases: The release list in target project
    :param studied_model:
    :param depend_model
    :param th:
    :param depend: Whether depending the results of LineDP
    :return:
    """

    cp_result_path = f'{root_path}Result/CP/{getattr(studied_model, "__name__")}_{th}'
    cp_depend_path = f'{root_path}Result/CP/{getattr(depend_model, "__name__")}_{th}'

    # ============================= Evaluating file level results ==================================
    # file-level result [oracle_list, prediction_list, mcc_list]
    file_level_performance = 'Setting,Test release,Recall,FAR,d2h,MCC\n'
    file_level_out_file = f'{cp_result_path}/cr_file_level_result_{proj_name}.pk'
    with open(file_level_out_file, 'rb') as file:
        data = pickle.load(file)
        file_level_performance += data[0] + data[1]

    save_csv_result(f'{cp_result_path}file_level_evaluation_{proj_name}.csv', file_level_performance)  # TODO 修改参数形式

    # ============================= Evaluating line level results ==================================
    line_level_performance = 'Setting,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for i in range(len(releases) - 1):
        test_release = releases[i + 1]
        print(f"Target release:\t{test_release} =========================="[:40])

        # line-level result [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict]
        line_level_out_file = f'{cp_result_path}/cr_line_level_result_{test_release}.pk'
        line_level_dep_file = f'{cp_depend_path}/cr_line_level_result_{test_release}.pk'

        try:
            with open(line_level_out_file, 'rb') as file:
                data = pickle.load(file)
                if depend:
                    print(f'Depend on {depend_model} in {line_level_dep_file}')
                    with open(line_level_dep_file, 'rb') as f:
                        dep_data = pickle.load(f)
                        data[3] = dep_data[3]

                line_level_performance += evaluation(test_release, data[0], data[1], data[2], data[3], data[4])
        except IOError:
            print(f'Error! Not found result file {line_level_out_file} or {line_level_dep_file}')
            return

    # Output the evaluation results for line level experiments
    save_csv_result(f'{cp_result_path}line_level_evaluation_{proj_name}.csv', line_level_performance)  # TODO 修改参数形式
