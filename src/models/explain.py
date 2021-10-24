# -*- coding:utf-8 -*-
import math

import pandas as pd

from src.models.base_model import BaseModel
from src.utils.helper import *
from src.utils.eval import evaluation

from lime.lime_text import LimeTextExplainer

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LineDP(BaseModel):
    model_name = 'LineDP'

    def __init__(self, train_release, test_release):
        super().__init__(train_release, test_release)

        # File level classifier
        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

    def file_level_prediction(self):
        print(f'{self.train_release}\t ===> \t{self.test_release}')
        # 2. Convert text feature into numerical feature, classifier
        # Neither perform lowercase, stemming, nor lemmatization. Remove tokens that appear only once
        train_vtr = self.vector.fit_transform(self.train_text)
        test_vtr = self.vector.transform(self.test_text)
        # 3. Predict defective files, 由依赖模型决定, 得到 test_predictions
        self.clf.fit(train_vtr, self.train_label)
        self.test_pred_labels = self.clf.predict(test_vtr)
        self.test_pred_scores = np.array([score[1] for score in self.clf.predict_proba(test_vtr)])

    def line_level_prediction(self):
        """
        :return: Ranking line-level defect-prone lines using Line_DP model. OK
        """
        print('Predicting line level defect prediction of LineDP')
        # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
        ranked_list_dict, worst_list_dict = {}, {}
        # Buggy lines 
        predicted_lines, predicted_score, predicted_density, total_lines = [], [], [], 0

        # Indices of defective files in descending order according to the prediction scores
        defective_file_index = [i for i in np.argsort(self.test_pred_scores)[::-1] if self.test_pred_labels[i] == 1]

        # Text tokenizer
        tokenizer = self.vector.build_tokenizer()
        c = make_pipeline(self.vector, self.clf)
        # Define an explainer
        explainer = LimeTextExplainer(class_names=['defect', 'non-defect'], random_state=self.random_seed)

        # Explain each defective file to predict the buggy lines exist in the file.
        # Process each file according to the order of defective rank list.
        for i in range(len(defective_file_index)):
            print(f'{i}/{len(defective_file_index)}')
            defective_filename = self.test_filename[defective_file_index[i]]
            # Some files are predicted as defective, but they are actually clean (i.e., FP files).
            # These FP files do not exist in the oracle. Therefore, the corresponding values of these files are []
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            # The code lines list of each corresponding predicted defective file
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]
            total_lines += len(defective_file_line_list)
            # ####################################### Core Section #################################################
            # Explain each defective file
            exp = explainer.explain_instance(' '.join(defective_file_line_list), c.predict_proba, num_features=100)
            # Extract top@20 risky tokens with positive scores. maybe less than 20
            risky_tokens = [x[0] for x in exp.as_list() if x[1] > 0][:20]

            # Count the number of risky tokens occur in each line.
            # The init value for each element of hit_count is [0 0 0 0 0 0 ... 0 0]. Note that line number index from 0.
            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                # Extract all tokens in the line with their original form.
                tokens_in_line = tokenizer(defective_file_line_list[line_index])
                # Check whether all risky tokens occurs in the line and count the number.
                for token in tokens_in_line:
                    if token in risky_tokens:
                        hit_count[line_index] += 1

            # ####################################### Core Section #################################################
            # Predicted buggy lines
            predicted_score.extend([hit_count[i] for i in range(num_of_lines) if hit_count[i] > 0])
            predicted_lines.extend([f'{defective_filename}:{i + 1}' for i in range(num_of_lines) if hit_count[i] > 0])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density.extend([density for i in range(num_of_lines) if hit_count[i] > 0])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.total_lines_in_defective_files = total_lines

        # Line level result
        data = {'predicted_buggy_lines': self.predicted_buggy_lines,
                'predicted_buggy_score': self.predicted_buggy_score,
                'predicted_density': self.predicted_density}
        data = pd.DataFrame(data, columns=['predicted_buggy_lines', 'predicted_buggy_score', 'predicted_density'])
        data.to_csv(self.line_level_result_file, index=False)


# OK 进行代码行级别的排序
def LineDP_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    """
    Ranking line-level defect-prone lines using Line_DP model
    :param proj:
    :param vector:
    :param classifier:
    :param test_text_lines:
    :param test_filename:
    :param test_predictions:
    :param out_file:
    :param threshold
    :return:
    """
    print('Predicting line level defect prediction of LineDP')
    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}

    oracle_line_list = []
    for file_name in oracle_line_dict:
        oracle_line_list.extend([f'{file_name}:{line}' for line in oracle_line_dict[file_name]])

    predicted_line_no = []
    predicted_line_score = []
    num_clean_lines = 0

    # Indices of defective files
    defective_file_indices = np.array([index[0] for index in np.argwhere(test_predictions > 0.5)])
    # Ranks of defective files
    defective_file_ranks = np.argsort([-x for x in test_predictions if x > 0.5])

    # Text tokenizer
    tokenizer = vector.build_tokenizer()
    c = make_pipeline(vector, classifier)
    # Define an explainer
    explainer = LimeTextExplainer(class_names=['defect', 'non-defect'], random_state=0)

    # Explain each defective file to predict the buggy lines exist in the file.
    # Process each file according to the order of defective rank list.
    for i in range(len(defective_file_indices)):
        defective_file_index = defective_file_indices[defective_file_ranks[i]]
        defective_file_name = test_filename[defective_file_index]

        # Some files are predicted as defective, but they are actually clean (i.e., FP files).
        # These FP files do not exist in the oracle_line_dict. Therefore, the corresponding values of these files are []
        if defective_file_name not in oracle_line_dict:
            oracle_line_dict[defective_file_name] = []
        # The code lines list of each corresponding predicted defective file
        defective_file_line_list = test_text_lines[defective_file_index]

        # ####################################### Core Section #################################################
        # 对分类结果进行解释
        exp = explainer.explain_instance(' '.join(defective_file_line_list),
                                         c.predict_proba,
                                         num_features=100,
                                         num_samples=5000)
        # Extract top20 risky tokens   maybe less than 20
        risky_tokens = [x[0] for x in exp.as_list() if x[1] > 0][:20]

        # Count the number of risky tokens occur in each line.
        # The init value for each element of hit_count is [0 0 0 0 0 0 ... 0 0]. Note that line number index from 0.
        hit_count = np.array([0] * len(defective_file_line_list))

        for line_index in range(len(defective_file_line_list)):
            # Extract all tokens in the line with their original form.
            tokens_in_line = tokenizer(defective_file_line_list[line_index])
            # Check whether all risky tokens occurs in the line and count the number.
            for risk_token in risky_tokens:
                if risk_token in tokens_in_line:
                    hit_count[line_index] += 1
        # ####################################### Core Section #################################################

        # Predicted buggy lines
        predicted_line_no.extend([f'{defective_file_name}:{i + 1}' for i in range(len(hit_count)) if hit_count[i] > 0])
        predicted_line_score.extend([hit_count[i] for i in range(len(hit_count)) if hit_count[i] > 0])
        num_clean_lines += len([i for i in hit_count if i > 0]) - len(oracle_line_dict[defective_file_name])

        # 根据命中次数对所有代码行进行降序排序, 按照排序后数值从大到小的顺序显示每个元素在原列表中的索引(i.e., 行号-1)
        # line + 1,因为原列表中代表行号的索引是从0开始计数而不是从1开始
        sorted_index = np.argsort(-hit_count)
        sorted_line_number = [line + 1 for line in sorted_index.tolist()]
        # 原始未经过调整的列表
        ranked_list_dict[defective_file_name] = sorted_line_number

        # ############################ Worst rank theoretically ###########################
        # 需要调整为最差排序的列表,当分数相同时
        worst_line_number = list(sorted_line_number)
        sorted_list = hit_count[sorted_index]
        worse_list, current_score, start_index, oracle_lines = [], -1, -1, oracle_line_dict[defective_file_name]
        for ii in range(len(sorted_list)):
            if sorted_list[ii] != current_score:
                current_score = sorted_list[ii]
                start_index = ii
            elif worst_line_number[ii] not in oracle_lines:
                temp = worst_line_number[ii]  # 取出这个无bug的行号
                for t in range(ii, start_index, -1):
                    worst_line_number[t] = worst_line_number[t - 1]
                worst_line_number[start_index] = temp
        worst_list_dict[defective_file_name] = worst_line_number
        # ############################ Worst rank theoretically ###########################
        print(f'{i}/{len(defective_file_indices)}')
    # Ranked buggy lines
    predicted_line_score = np.array(predicted_line_score)
    predicted_line_no = np.array(predicted_line_no)
    sorted_line_no = predicted_line_no[np.argsort(-predicted_line_score)]

    # Evaluation
    tp, fp, fn, tn, x, y, n, N = .0, .0, .0, .0, .0, .0, .0, .0
    for buggy_line_no in sorted_line_no:
        if buggy_line_no in oracle_line_list:
            tp += 1
        else:
            fp += 1
    fn = len(oracle_line_list) - tp
    tn = num_clean_lines - fn
    x = len(sorted_line_no)
    y = tp
    n = len(oracle_line_list)
    N = tp + fp + fn + tn

    print(tp, fp, tn, fn)

    # Classification indicators
    precision = .0 if tp + fp == .0 else tp / (tp + fp)
    ce = .0 if fn + tn == .0 else fn / (fn + tn)
    recall = .0 if tp + fn == .0 else tp / (tp + fn)
    far = .0 if fp + tn == 0 else fp / (fp + tn)
    d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(0 - far, 2)) / math.sqrt(2)
    mcc = .0 if tp + fp == .0 or tp + fn == .0 or tn + fp == .0 or tn + fn == .0 else \
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    er = (y * N - x * n) / (y * N)
    ri = (y * N - x * n) / (x * n)

    # Ranking indicators
    ifa = 0
    for buggy_line_no in sorted_line_no:
        if buggy_line_no not in oracle_line_list:
            ifa += 1
        else:
            break
    r20, buggy_lines, effort, max_effort = .0, 0, 0, int(N * 0.2)
    for buggy_line_no in sorted_line_no:
        if effort > max_effort:
            break
        effort += 1
        if buggy_line_no in oracle_line_list:
            buggy_lines += 1

    r20 = buggy_lines / len(oracle_line_list)
    print('Precision, CE, Recall, FAR, D2H, MCC, ER, RI, R@20%, IFA')
    print(f'{precision}, {ce}, {recall}, {far}, {d2h}, {mcc}, {er}, {ri}, {r20},{ifa}')


# OK 进行代码行级别的排序
def TMI_LR_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    """
    Ranking line-level defect-prone lines using TMI-LR model
    :param proj:
    :param vector:
    :param classifier:
    :param test_text_lines:
    :param test_filename:
    :param test_predictions:
    :param out_file:
    :param threshold
    :return:
    """
    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}
    defect_cf_dict = {}
    effort_cf_dict = {}

    # 预测为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])
    # 文本分词器
    tokenizer = vector.build_tokenizer()

    # 标准化处理
    std = StandardScaler()
    std_coefficient = std.fit_transform(classifier.coef_.reshape(-1, 1))
    # 特征重要性字典
    feature_weight_dict = dict(zip(vector.get_feature_names(), std_coefficient.T[0]))
    # 按照重要性排序后的元祖列表
    sorted_feature_weight_dict = sorted(feature_weight_dict.items(), key=lambda kv: (-kv[1], kv[0]))

    # 对预测为有bug的文件逐个进行解释结果来进行代码行级别的预测
    for i in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[i]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ####################################### 核心部分 #################################################
        # 对分类结果进行解释
        # 取出risk tokens, 取前20个, 可能不足20个
        positive_tokens = [x[0] for x in sorted_feature_weight_dict if x[1] > 0][:20]

        # 统计 每一行中出现 risk tokens 的个数, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数
        hit_count = np.array([0] * len(target_file_lines))
        for line_index in range(len(target_file_lines)):
            # 取出该行中的所有单词, 保留其原始形态
            tokens_in_line = tokenizer(target_file_lines[line_index])
            # 检测所有 risk tokens 是否在该行中出现, 并且统计出现的个数
            for risk_token in positive_tokens:
                if risk_token in tokens_in_line:
                    hit_count[line_index] += 1
        # ####################################### 核心部分 #################################################

        # 根据命中次数对所有代码行进行降序排序, 按照排序后数值从大到小的顺序显示每个元素在原列表中的索引(i.e., 行号-1)
        # line + 1,因为原列表中代表行号的索引是从0开始计数而不是从1开始
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
        # 20% effort (i.e, LOC)
        effort_cf_dict[target_file_name] = int(.2 * len(target_file_lines))
        # 设置分类切分点: 所有包含risk tokens (i.e., hit_count[i] > 0) 的代码行被预测为有 bug
        defect_cf_dict[target_file_name] = len([hit for hit in hit_count if hit > 0])
        print('%d/%d files predicted finish!' % (i, len(defect_prone_file_indices)))

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)


# OK 进行代码行级别的排序
def TMI_SVM_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return TMI_LR_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold)


# OK 进行代码行级别的排序
def TMI_MNB_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return TMI_LR_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold)


# OK 进行代码行级别的排序
def TMI_RF_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    """
    Ranking line-level defect-prone lines using Line_DP model
    :param proj:
    :param vector:
    :param classifier:
    :param test_text_lines:
    :param test_filename:
    :param test_predictions:
    :param out_file:
    :param threshold
    :return:
    """
    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}
    defect_cf_dict = {}
    effort_cf_dict = {}

    # 预测为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])
    # 文本分词器
    tokenizer = vector.build_tokenizer()

    # 特征重要性字典
    feature_weight_dict = dict(zip(vector.get_feature_names(), classifier.feature_importances_.tolist()))
    # 按照重要性排序后的元祖列表
    sorted_feature_weight_dict = sorted(feature_weight_dict.items(), key=lambda kv: (-kv[1], kv[0]))

    # 对预测为有bug的文件逐个进行解释结果来进行代码行级别的预测
    for i in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[i]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ####################################### 核心部分 #################################################
        # 对分类结果进行解释
        # 取出risk tokens, 取前20个, 可能不足20个
        positive_tokens = [x[0] for x in sorted_feature_weight_dict if x[1] > 0][:20]

        # 统计 每一行中出现 risk tokens 的个数, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数
        hit_count = np.array([0] * len(target_file_lines))
        for line_index in range(len(target_file_lines)):
            # 取出该行中的所有单词, 保留其原始形态
            tokens_in_line = tokenizer(target_file_lines[line_index])
            # 检测所有 risk tokens 是否在该行中出现, 并且统计出现的个数
            for risk_token in positive_tokens:
                if risk_token in tokens_in_line:
                    hit_count[line_index] += 1
        # ####################################### 核心部分 #################################################

        # 根据命中次数对所有代码行进行降序排序, 按照排序后数值从大到小的顺序显示每个元素在原列表中的索引(i.e., 行号-1)
        # line + 1,因为原列表中代表行号的索引是从0开始计数而不是从1开始
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
        # 20% effort (i.e, LOC)
        effort_cf_dict[target_file_name] = int(.2 * len(target_file_lines))
        # 设置分类切分点: 所有包含risk tokens (i.e., hit_count[i] > 0) 的代码行被预测为有 bug
        defect_cf_dict[target_file_name] = len([hit for hit in hit_count if hit > 0])
        print('%d/%d files predicted finish!' % (i, len(defect_prone_file_indices)))

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)


# OK 进行代码行级别的排序
def TMI_DT_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return TMI_RF_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold)
