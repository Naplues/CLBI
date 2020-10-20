# -*- coding:utf-8 -*-

import numpy as np
from src.utils.helper import *
from src.utils.eval import evaluation


def call_number(statement):
    statement = statement.strip('\"')
    score = 0
    for char in statement:
        if char == '(':
            score += 1
    return score


# 进行代码行级别的排序
def AccessModel(proj, vector, clf, test_text_lines, test_filename, test_predictions, out_file, threshold):
    """
    Line-level ranking
    :param proj:
    :param vector:
    :param clf
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

    # 预测值为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])
    # 文本分词器
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
            if len(tokens_in_line) == 0:
                hit_count[index] = 0
            else:
                hit_count[index] = len(tokens_in_line) * call_number(target_file_lines[index]) + 1

            weight = 2

            if 'for' in tokens_in_line:
                hit_count[index] *= weight
            if 'while' in tokens_in_line:
                hit_count[index] *= weight
            if 'do' in tokens_in_line:
                hit_count[index] *= weight

            if 'if' in tokens_in_line:
                hit_count[index] *= weight
            if 'else' in tokens_in_line:
                hit_count[index] *= weight
            if 'switch' in tokens_in_line:
                hit_count[index] *= weight
            if 'case' in tokens_in_line:
                hit_count[index] *= weight

            if 'continue' in tokens_in_line:
                hit_count[index] *= weight
            if 'break' in tokens_in_line:
                hit_count[index] *= weight
            if 'return' in tokens_in_line:
                hit_count[index] *= weight

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
        defect_cf_dict[target_file_name] = int(threshold / 100.0 * len(target_file_lines))

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)
