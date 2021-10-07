# -*- coding:utf-8 -*-
import math
import warnings
import numpy as np

from src.models.access import *
from src.models.explain import LineDP
from src.models.natural import LM_2_Model
from src.models.static_analysis_tools import *
from src.utils.helper import *
from sklearn import metrics
from src.utils.eval import evaluation

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)


def predict_cross_release(proj_name, releases, studied_model, depend_model=None, th=50):
    """
    跨版本实验: 训练 Vi 预测 Vi+1
    Predict the results under cross release experiments: Training release i to predict test release i + 1
    :param proj_name: Target project name (i.e., Ambari)
    :param releases: Studied releases (i.e., [Ambari-2.1.0, Ambari-2.2.0, Ambari-2.4.0])
    :param studied_model: The prediction model (i.e, AccessModel)
    :param depend_model: The depend model (i.e., LineDPModel)
    :param th:
    :return:
    """
    print(f'========== Cross-release prediction for {proj_name} =============================================='[:60])
    model_name = getattr(studied_model, "__name__")  # A返回调用方法的属性值，如模型函数的名字属性 AccessModel
    # 某个预测模型的跨版本实验结果存储路径. 如, D:/CLDP_data/Result/CP/AccessModel_50/
    cp_result_path = f'{root_path}Result/CP/{model_name}_{th}/'
    make_path(cp_result_path)

    print("Training set\t ===> \tTest set.")
    for i in range(len(releases) - 1):
        # 1. Loading data. train data index = i, test data index = i + 1
        train_release, test_release = releases[i], releases[i + 1]

        model = LineDP(train_release, test_release)
        model.file_level_prediction()

        # 4. 储存文件级别的预测结果和评估指标
        model.analyze_file_level_result()

        model.line_level_prediction()

        # out_file = f'{cp_result_path}cr_line_level_result_{test_release}.pk'
        # studied_model(test_release, vector, clf, test_text_lines, test_filename, test_prediction_scores, out_file, th)
    #
    # dump_pk_result(f'{cp_result_path}cr_file_level_result_{proj_name}.pk', [oracle_list, prediction_label_list, mcc_list])
    # print('Avg MCC:\t%.3f\n' % np.average(mcc_list))


"""
        # 如果未指定依赖模型, 依赖模型就是预测模型本身,依赖模型用于文件级别的预测
        depend_model = studied_model if depend_model is None else depend_model
        
        if depend_model == AccessModel \
                or depend_model == NFC_Model or depend_model == NT_Model \
                or depend_model == PMDModel or depend_model == CheckStyleModel \
                or depend_model == LM_2_Model:
            # Unsupervised models
            # print('File level classifier: Effort-aware ManualDown')
            file_level_out_file = f'{cp_result_path}/cr_file_level_result_{proj_name}.pk'
            if not os.path.exists(file_level_out_file):
                test_prediction_labels = EAMD(test_release, test_text, test_text_lines, test_labels)
            else:
                with open(file_level_out_file, 'rb') as file:
                    test_prediction_labels = pickle.load(file)[1][i]
        else:
            # Supervised models default parameter settings
            # Define MIs-based file-level classifier
            if depend_model == TMI_LR_Model:
                print('File level classifier: Logistic')
                clf = LogisticRegression(random_state=0).fit(train_vtr, train_label)
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
                clf = LogisticRegression(random_state=0).fit(train_vtr, train_label)


"""


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

    cp_result_path = f'{root_path}Result/CP/{getattr(studied_model, "__name__")}_{th}/'
    cp_depend_path = f'{root_path}Result/CP/{getattr(depend_model, "__name__")}_{th}/'

    # ============================= Evaluating file level results ==================================
    # file-level result [oracle_list, prediction_list, mcc_list]
    file_level_performance = 'Release,TP,FP,TN,FN,Precision,Recall,F1,MCC\n'
    file_level_out_file = f'{cp_result_path}cr_file_level_result_{proj_name}.pk'
    with open(file_level_out_file, 'rb') as file:
        data = pickle.load(file)
        # 处理每个测试版本
        for i in range(len(releases) - 1):
            test_release = releases[i + 1]
            oracle, prediction, MCC = data[0][i], data[1][i], data[2][i]
            tp, fp, tn, fn = 0, 0, 0, 0
            for index in range(len(oracle)):
                if prediction[index] == 1 and oracle[index] == 1:
                    tp += 1
                elif prediction[index] == 1 and oracle[index] == 0:
                    fp += 1
                elif prediction[index] == 0 and oracle[index] == 0:
                    tn += 1
                else:
                    fn += 1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            # adjust display format
            precision, recall, f1, mcc = round(precision, 3), round(recall, 3), round(f1, 3), round(mcc, 3)
            file_level_performance += f'{test_release},{tp},{fp},{tn},{fn},{precision},{recall},{f1},{mcc}\n'

    save_csv_result(cp_result_path, f'file_level_evaluation_{proj_name}.csv', file_level_performance)

    # ============================= Evaluating line level results ==================================
    line_level_performance = 'Setting,Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median\n'
    for i in range(len(releases) - 1):
        test_release = releases[i + 1]
        # print(f"Target release:\t{test_release} =========================="[:40])

        # line-level result [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict]
        line_level_out_file = f'{cp_result_path}cr_line_level_result_{test_release}.pk'
        line_level_dep_file = f'{cp_depend_path}cr_line_level_result_{test_release}.pk'

        try:
            with open(line_level_out_file, 'rb') as file:
                data = pickle.load(file)

                if depend:
                    print(f'Depend on {depend_model} in {line_level_dep_file}')
                    with open(line_level_dep_file, 'rb') as f:
                        dep_data = pickle.load(f)
                        data[3] = dep_data[3]
                        print(len(dep_data))

                line_level_performance += evaluation(test_release, data[0], data[1], data[2], data[3], data[4])
        except IOError:
            print(f'Error! Not found result file {line_level_out_file} or {line_level_dep_file}')
            return

    # Output the evaluation results for line level experiments
    save_csv_result(cp_result_path, f'line_level_evaluation_{proj_name}.csv', line_level_performance)
