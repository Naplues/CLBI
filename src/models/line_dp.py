# -*- coding:utf-8 -*-

from src.utils.helper import *
from src.utils.eval import evaluation
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

random_seed = 0  # random seed is set as 0-9


# OK 进行代码行级别的排序
def LineDPModel(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
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

    # 制作管道
    c = make_pipeline(vector, classifier)
    # 定义解释器
    explainer = LimeTextExplainer(class_names=['defect', 'non-defect'], random_state=random_seed)

    # 对预测为有bug的文件逐个进行解释结果来进行代码行级别的预测
    for defect_file_index in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[defect_file_index]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ####################################### 核心部分 #################################################
        # 对分类结果进行解释
        exp = explainer.explain_instance(' '.join(target_file_lines), c.predict_proba, num_features=100)
        # 取出risk tokens, 取前20个, 可能不足20个
        positive_tokens = [x[0] for x in exp.as_list() if x[1] > 0][:20]

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
        print('%d/%d files predicted finish!' % (defect_file_index, len(defect_prone_file_indices)))

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)
