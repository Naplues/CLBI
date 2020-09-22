# -*- coding:utf-8 -*-

import math
import numpy as np


# 评估行级别的分类效果
def evaluation(proj, oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict):
    """
    评估行级别的分类及排序效果
    以下四个字典类型的变量均以文件名作为 key
    :param proj: 项目版本名
    :param oracle_line_dict: 真实的 bug行列表字典
    :param ranked_list_dict: 预测的 bug行序列字典
    :param defect_cut_off_dict: 二分类 切分点字典
    :param effort_cut_off_dict: 工作量 切分点字典
    :return:
    """
    # 预测为有bug的行号
    predict_as_bug_line_dict = {}
    # 预测为无bug的行号
    predict_as_clean_line_dict = {}

    # ################## 按照二分类进行切分 工作量感知的指标  Recall, FAR, d2h, MCC ########################
    for target_file_name, ranked_list in ranked_list_dict.items():
        cut_off = defect_cut_off_dict[target_file_name]
        predict_as_bug_line_dict[target_file_name] = ranked_list[:cut_off]
        predict_as_clean_line_dict[target_file_name] = ranked_list[cut_off:]
    # 混淆矩阵
    tp, fp, tn, fn = .0, .0, .0, .0
    # 计算评估指标
    # 统计出预测为有bug的代码行实例 TP, FP
    for filename, predicted_line_numbers in predict_as_bug_line_dict.items():
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中
        if filename not in oracle_line_dict:
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number in oracle_line_numbers:
                tp += 1
            else:
                fp += 1

    # 统计出预测为无bug的代码行实例 TN, FN
    for filename, predicted_line_numbers in predict_as_clean_line_dict.items():
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中
        if filename not in oracle_line_dict:
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number in oracle_line_numbers:
                fn += 1
            else:
                tn += 1

    # 计算指标
    recall = tp / (tp + fn)
    far = fp / (fp + tn)
    d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(0 - far, 2)) / math.sqrt(2)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # ################## 按照20%工作量进行切分 工作量感知的排序指标 Recall@20% ########################
    for target_file_name, ranked_list in ranked_list_dict.items():
        cut_off = effort_cut_off_dict[target_file_name]
        predict_as_bug_line_dict[target_file_name] = ranked_list[:cut_off]
        predict_as_clean_line_dict[target_file_name] = ranked_list[cut_off:]
    # 混淆矩阵
    tp, fp, tn, fn = .0, .0, .0, .0
    # 计算评估指标
    # 统计出预测为有bug的实例 TP, FP
    for filename, predicted_line_numbers in predict_as_bug_line_dict.items():
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中
        if filename not in oracle_line_dict:
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number in oracle_line_numbers:
                tp += 1
            else:
                fp += 1

    # 统计出预测为无bug的实例 TN, FN
    for filename, predicted_line_numbers in predict_as_clean_line_dict.items():
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中
        if filename not in oracle_line_dict:
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number in oracle_line_numbers:
                fn += 1
            else:
                tn += 1

    recall_20 = tp / (tp + fn)

    # 统计IFA
    ifa_list = []
    for filename, predicted_line_numbers in ranked_list_dict.items():
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中
        if filename not in oracle_line_dict:
            continue
        ifa = 0
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number not in oracle_line_numbers:
                ifa += 1
            else:
                break
        ifa_list.append(ifa)
    ifa_mean = int(np.mean(ifa_list))
    ifa_median = int(np.median(ifa_list))
    ifa = ','.join([str(i) for i in ifa_list])

    # ################################# 计算排序指标 MRR MAP ###########################################
    _mrr, _map, n = .0, .0, .0
    for filename, predicted_line_numbers in ranked_list_dict.items():
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中
        if filename not in oracle_line_dict:
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        rr, ap, i, k = .0, .0, 0, len(oracle_line_numbers)
        n += 1
        for index in range(len(predicted_line_numbers)):
            line_number = predicted_line_numbers[index]
            if line_number in oracle_line_numbers:
                rr = 1. / (index + 1)
                break
        _mrr += rr
        for index in range(len(predicted_line_numbers)):
            line_number = predicted_line_numbers[index]
            if line_number in oracle_line_numbers:
                i += 1
                ap += float(i) / (index + 1)
                if i == k:
                    break
        _map += ap / k

    _mrr /= n
    _map /= n

    print('recall\tFAR\td2h\tMCC\tr_20%\tIFA_avg\tIFA_med\tMRR\tMAP')
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%.3f\t%.3f\n' %
          (recall, far, d2h, mcc, recall_20, ifa_mean, ifa_median, _mrr, _map))

    return '%s,%f,%f,%f,%f,%f,%d,%d,%f,%f,%s\n' % (
        proj, recall, far, d2h, mcc, recall_20, ifa_mean, ifa_median, _mrr, _map, ifa)
