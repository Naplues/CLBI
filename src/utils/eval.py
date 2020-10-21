# -*- coding:utf-8 -*-

import math
import numpy as np
from src.utils.helper import *


# 评估行级别的分类效果
def evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cut_off_dict, effort_cut_off_dict):
    """
    评估行级别的分类及排序效果
    以下四个字典类型的变量均以文件名作为 key
    :param proj: 项目版本名
    :param oracle_line_dict: 真实的 bug行列表字典
    :param ranked_list_dict: 预测的 bug行序列字典
    :param worst_list_dict:  预测的 bug行最差序列字典
    :param defect_cut_off_dict: 二分类 切分点字典
    :param effort_cut_off_dict: 工作量 切分点字典
    :return:
    """
    r_normal = 'normal,' + evaluator(proj, oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict)
    r_worst = 'worst,' + evaluator(proj, oracle_line_dict, worst_list_dict, defect_cut_off_dict, effort_cut_off_dict)
    return f'{r_normal}{r_worst}'


# 评估行级别的分类效果
def evaluator(proj, oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict):
    # 预测为有bug的行号
    predict_as_bug_line_dict = {}
    # 预测为无bug的行号
    predict_as_clean_line_dict = {}

    tp_dict = []
    rank_dict = {}
    # ################## 按照二分类进行切分 工作量感知的指标  Recall, FAR, d2h, MCC ########################
    for target_file_name, ranked_list in ranked_list_dict.items():
        cut_off = defect_cut_off_dict[target_file_name]
        predict_as_bug_line_dict[target_file_name] = ranked_list[:cut_off]
        predict_as_clean_line_dict[target_file_name] = ranked_list[cut_off:]
    # 混淆矩阵
    tp, fp, tn, fn = .0, .0, .0, .0
    # 计算评估指标
    # 统计出预测为有bug的代码行实例 TP, FP
    for filename, predicted_line_ranks in predict_as_bug_line_dict.items():
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_ranks:
            if line_number in oracle_line_numbers:
                tp += 1
                tp_dict.append(filename + ':' + str(line_number))
            else:
                fp += 1

    # 统计出预测为无bug的代码行实例 TN, FN
    for filename, predicted_line_ranks in predict_as_clean_line_dict.items():
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_ranks:
            if line_number in oracle_line_numbers:
                fn += 1
            else:
                tn += 1

    # 计算指标
    recall = .0 if tp + fn == .0 else tp / (tp + fn)
    far = .0 if fp + tn == 0 else fp / (fp + tn)
    d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(0 - far, 2)) / math.sqrt(2)

    mcc = .0 if tp + fp == .0 or tp + fn == .0 or tn + fp == .0 or tn + fn == .0 else \
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    ce = .0 if fn + tn == .0 else fn / (fn + tn)

    # ################## 按照20%工作量进行切分 工作量感知的排序指标 Recall@20% ########################
    for target_file_name, ranked_list in ranked_list_dict.items():
        cut_off = effort_cut_off_dict[target_file_name]
        predict_as_bug_line_dict[target_file_name] = ranked_list[:cut_off]
        predict_as_clean_line_dict[target_file_name] = ranked_list[cut_off:]
    # 混淆矩阵
    tp, fp, tn, fn = .0, .0, .0, .0
    # 计算评估指标
    # 统计出预测为有bug的实例 TP, FP
    for filename, predicted_line_ranks in predict_as_bug_line_dict.items():
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_ranks:
            if line_number in oracle_line_numbers:
                tp += 1
            else:
                fp += 1

    # 统计出预测为无bug的实例 TN, FN
    for filename, predicted_line_ranks in predict_as_clean_line_dict.items():
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_ranks:
            if line_number in oracle_line_numbers:
                fn += 1
            else:
                tn += 1

    recall_20 = .0
    if tp + fn != .0:
        recall_20 = tp / (tp + fn)

    # 统计IFA
    ifa_list = []
    for filename, predicted_line_ranks in ranked_list_dict.items():
        ifa = 0
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_ranks:
            if line_number not in oracle_line_numbers:
                ifa += 1
            else:
                break
        ifa_list.append(ifa)

    ifa_mean = -1 if len(ifa_list) == 0 else int(np.mean(ifa_list))
    ifa_median = -1 if len(ifa_list) == 0 else int(np.median(ifa_list))
    ifa = ','.join([str(i) for i in ifa_list])

    # ################################# 计算排序指标 MRR MAP ###########################################
    _mrr, _map, n = .0, .0, .0
    for filename, predicted_line_ranks in ranked_list_dict.items():
        oracle_line_numbers = oracle_line_dict[filename]

        for index in range(len(predicted_line_ranks)):
            line_number = predicted_line_ranks[index]
            if line_number in oracle_line_numbers:
                rank_dict[filename + ':' + str(line_number)] = index

        rr, ap, i, k = .0, .0, 0, len(oracle_line_numbers)
        n += 1
        if k == 0:
            continue
        for index in range(len(predicted_line_ranks)):
            line_number = predicted_line_ranks[index]
            if line_number in oracle_line_numbers:
                rr = 1. / (index + 1)
                break
        _mrr += rr
        for index in range(len(predicted_line_ranks)):
            line_number = predicted_line_ranks[index]
            if line_number in oracle_line_numbers:
                i += 1
                ap += float(i) / (index + 1)
                if i == k:
                    break
        _map += ap / k

    _mrr = -1 if n == 0 else _mrr / n
    _map = -1 if n == 0 else _map / n

    # Dump Classification Diff
    # dump_pk_result(result_path + 'Diff_Classification/' + proj + '.pk', tp_dict)
    dump_pk_result(result_path + 'Diff_Ranking/' + proj + '.pk', rank_dict)
    print('recall\tFAR\td2h\tMCC\tCE\tr_20%\tIFA_avg\tIFA_med')
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\n' % (recall, far, d2h, mcc, ce, recall_20, ifa_mean, ifa_median))

    return f'{proj},{recall},{far},{d2h},{mcc},{ce},{recall_20},{ifa_mean},{ifa_median}\n'
