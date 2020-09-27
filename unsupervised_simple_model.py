# -*- coding:utf-8 -*-

import warnings
from helper import *
from sklearn import metrics
from evaluation import evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

# 全局变量设置

root_path = r'C://Users/GZQ/Desktop/CLDP_data'
file_level_path = root_path + '/Dataset/File-level/'
line_level_path = root_path + '/Dataset/Line-level/'
result_path = root_path + '/Result/Unsupervised/'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'

make_path(result_path)


# 版本内预测实验
def within_release_prediction(proj, num_iter=10, num_folds=10):
    """
    版本内预测
    :param proj: 项目版本名
    :param num_iter: 重复次数 默认10
    :param num_folds:折数 默认 10
    :return:
    """

    log = '=' * 10 + ' Within-release prediction for ' + proj + ' ' + '=' * 60
    print(log[:60])
    # 声明储存预测结果变量
    test_list = []
    prediction_list = []
    # 声明储存评估指标变量
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []

    # Line-level指标
    performance = 'Iter-Fold,Recall,FAR,d2h,MCC,Recall@20%,IFA_mean,IFA_median\n'
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
            fold += 1
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
            test_list.append(test_labels)
            prediction_list.append(test_predictions)
            precision_list.append(metrics.precision_score(test_labels, test_predictions))
            recall_list.append(metrics.recall_score(test_labels, test_predictions))
            f1_list.append(metrics.f1_score(test_labels, test_predictions))
            mcc_list.append(metrics.matthews_corrcoef(test_labels, test_predictions))

            # 5. 解释代码行级别的缺陷概率
            out_file = result_path + 'with_predictions_' + proj + str(it) + '_line_risk_ranks.pk'
            simple(proj, test_text_lines, test_filename, test_predictions, out_file)

    # 将行级别的性能评估结果写入文件
    with open(result_path + 'wr_line_level_evaluation_' + proj + '.csv', 'w') as file:
        file.write(performance)

    # 打印平均结果
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
    prediction_list = []
    # 声明储存评估指标变量
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []

    # Line-level指标
    performance = 'Test release,Recall,FAR,d2h,MCC,CE,Recall@20%,IFA_mean,IFA_median,MRR,MAP,IFA list\n'
    print("Training set\t ===> \tTest set.")
    for i in range(len(releases_list) - 1):
        # 1. 读取数据 训练版本的索引为 i, 测试版本的索引为 i + 1
        train_proj, test_proj = releases_list[i], releases_list[i + 1]
        print("%s\t ===> \t%s" % (train_proj, test_proj))
        #    源码文本列表 源码文本行级别列表 标签列表 文件名称
        test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(test_proj)

        # 先排序后预测
        size_list = np.array([len(test_instance) for test_instance in test_text_lines])
        sorted_index = np.argsort(-size_list)
        cut_off = int(.5 * len(sorted_index))
        buggy_file_index = sorted_index[:cut_off]
        clean_file_index = sorted_index[cut_off:]

        test_predictions = []
        for index in range(len(test_text)):
            if index in buggy_file_index:
                test_predictions.append(1)
            if index in clean_file_index:
                test_predictions.append(0)

        # 4. 储存文件级别的预测结果和评估指标
        test_list.append(test_label)
        prediction_list.append(test_predictions)
        precision_list.append(metrics.precision_score(test_label, test_predictions))
        recall_list.append(metrics.recall_score(test_label, test_predictions))
        f1_list.append(metrics.f1_score(test_label, test_predictions))
        mcc_list.append(metrics.matthews_corrcoef(test_label, test_predictions))

        # 5. 解释代码行级别的缺陷概率
        out_file = result_path + 'cr_line_level_ranks_' + test_proj + '.pk'

        # 如果模型的结果已经存在直接进行评估, 否则重新进行预测并评估
        if os.path.exists(out_file):
            with open(out_file, 'rb') as file:
                data = pickle.load(file)
                # with open(root_path + '/Result/LineDP/cr_line_level_ranks_' + test_proj + '.pk',  'rb') as f:
                # cut_data = pickle.load(f) cut_data[1], cut_data[2]
                performance += evaluation(proj, data[0], data[1], data[2], data[3], data[4])
        else:
            pass
    # performance += simple(test_proj, test_text_lines, test_filename, test_predictions, out_file)

    # 输出行级别的结果
    with open(result_path + 'cr_line_level_evaluation_' + proj + '.csv', 'w') as file:
        file.write(performance)

    # 打印文件级别的平均结果
    print('Avg MCC:\t%.3f\n' % np.average(mcc_list))
    with open(result_path + 'cr_file_level_evaluation_' + proj + '.pk', 'wb') as file:
        pickle.dump([test_list, prediction_list, precision_list, recall_list, f1_list, mcc_list], file)


# 进行代码行级别的排序
def simple(proj, test_text_lines, test_filename, test_predictions, out_file):
    """
    Line-level ranking
    :param proj:
    :param vector:
    :param test_text_lines:
    :param test_filename:
    :param test_predictions:
    :param out_file:
    :return:
    """
    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}
    defect_cf_dict = {}
    effort_cf_dict = {}

    # 预测值为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])
    # 所有的测试文件
    # defect_prone_file_indices = np.array([index for index in range(len(test_predictions))])

    # 文本分词器
    # 2. 定义一个矢量器
    vector = CountVectorizer(lowercase=False, min_df=2)
    tokenizer = vector.build_tokenizer()

    # 对预测为有bug的文件逐个进行代码行级别的排序
    for i in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[i]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的测试文件(被预测为有bug,但实际上)没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ############################ 重点,怎么给每行赋一个缺陷值 ################################
        # 计算 每一行的权重, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数
        hit_count = np.array([.0] * len(target_file_lines))
        for index in range(len(target_file_lines)):
            tokens_in_line = tokenizer(target_file_lines[index])
            #  考虑调用深度和单词数
            hit_count[index] = len(tokens_in_line) * call_depth(target_file_lines[index])
            if 'return' in tokens_in_line:
                hit_count[index] += len(tokens_in_line) * 1  # 1
        # ############################ 重点,怎么给每行赋一个缺陷值 ################################

        # 根据命中次数对代码行进行降序排序, 按照排序后数值从大到小的顺序显示代码行在原列表中的索引, cut_off 为切分点
        # line + 1,因为下标是从0开始计数而不是从1开始
        sorted_index = np.argsort(-hit_count)
        sorted_line_number = [line + 1 for line in sorted_index.tolist()]
        # 原始未经过调整的列表
        ranked_list_dict[target_file_name] = sorted_line_number

        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################
        # 需要调整为最差排序的列表,当分数相同时
        worst_line_number = list(sorted_line_number)
        sorted_list = hit_count[sorted_index]
        worse_list, current_score, start_index, oracle_lines = [], -1, -1, oracle_line_dict[target_file_name]
        for ii in range(len(sorted_list)):
            if sorted_list[ii] != current_score:
                current_score = sorted_list[ii]
                start_index = ii
            elif worst_line_number[ii] not in oracle_lines:
                temp = worst_line_number[ii]  # 取出这个无bug的行号
                for t in range(ii, start_index, -1):
                    worst_line_number[t] = worst_line_number[t - 1]
                worst_line_number[start_index] = temp
        worst_list_dict[target_file_name] = worst_line_number
        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################

        # ###################################### 切分点设置 ################################################
        # 20% effort (i.e., LOC)
        effort_cf_dict[target_file_name] = int(.2 * len(target_file_lines))
        # 设置分类切分点: 默认前50%
        defect_cf_dict[target_file_name] = int(.5 * len(target_file_lines))

    dump_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)


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

    # 组合评估结果文件
    combine_results(result_path)


if __name__ == '__main__':
    # 运行版本内预测实验
    # run_within_release_prediction()
    # 运行版本间预测实验
    run_cross_release_prediction()
