# -*- coding:utf-8 -*-
import math
import numpy as np


# ################################## Evaluation #######################################
# 评估行级别的分类效果
def evaluation(oracle_line_dict, ranked_list_dict, defect_cut_off_dict, effort_cut_off_dict):
    # predict_as_bug_line_dict: 存储每个文件中被预测为有bug的代码行号 predict_as_bug_line_dict[filename] = [line numbers]
    predict_as_bug_line_dict = {}
    predict_as_clean_line_dict = {}

    # ################## 工作量感知的指标 Recall, FAR, d2h, MCC ########################
    for target_file_name, ranked_list in ranked_list_dict.items():
        cut_off = defect_cut_off_dict[target_file_name]
        predict_as_bug_line_dict[target_file_name] = ranked_list[:cut_off]
        predict_as_clean_line_dict[target_file_name] = ranked_list[cut_off:]
    # 混淆矩阵
    tp, fp, tn, fn = .0, .0, .0, .0
    # 计算评估指标
    # 统计出预测为有bug的实例 TP, FP
    for filename, predicted_line_numbers in predict_as_bug_line_dict.items():
        # 这句应该不会执行, 因为素有的filename应该都在 oracle_line_dict 中
        if filename not in oracle_line_dict:
            # print('test %s' % filename)
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number in oracle_line_numbers:
                tp += 1
            else:
                fp += 1

    # 统计出预测为无bug的实例 TN, FN
    for filename, predicted_line_numbers in predict_as_clean_line_dict.items():
        # 这句应该不会执行, 因为素有的filename应该都在 oracle_line_dict 中
        if filename not in oracle_line_dict:
            # print('test %s' % filename)
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
    d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(far, 2)) / math.sqrt(2)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # ################## 工作量感知的排序指标 Recall@20% ########################
    for target_file_name, ranked_list in ranked_list_dict.items():
        cut_off = effort_cut_off_dict[target_file_name]
        predict_as_bug_line_dict[target_file_name] = ranked_list[:cut_off]
        predict_as_clean_line_dict[target_file_name] = ranked_list[cut_off:]
    # 混淆矩阵
    tp, fp, tn, fn = .0, .0, .0, .0
    # 计算评估指标
    # 统计出预测为有bug的实例 TP, FP
    for filename, predicted_line_numbers in predict_as_bug_line_dict.items():
        # 这句应该不会执行, 因为素有的filename应该都在 oracle_line_dict 中
        if filename not in oracle_line_dict:
            # print('test %s' % filename)
            continue
        oracle_line_numbers = oracle_line_dict[filename]
        for line_number in predicted_line_numbers:
            if line_number in oracle_line_numbers:
                tp += 1
            else:
                fp += 1

    # 统计出预测为无bug的实例 TN, FN
    for filename, predicted_line_numbers in predict_as_clean_line_dict.items():
        # 这句应该不会执行, 因为素有的filename应该都在 oracle_line_dict 中
        if filename not in oracle_line_dict:
            # print('test %s' % filename)
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
        # 这句应该不会执行, 因为素有的filename应该都在 oracle_line_dict 中
        if filename not in oracle_line_dict:
            # print('test %s' % filename)
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
    print('recall,\tFAR,\td2h,\tMCC\n%.3f\t%.3f\t%.3f\t%.3f\n' % (recall, far, d2h, mcc))
    print('recall_20%%,\t mean IFA,\t median IFA\n%.3f\t%d\t%d\n' % (recall_20, ifa_mean, ifa_median))

    return '%f,%f,%f,%f,%f,%d,%d\n' % (recall, far, d2h, mcc, recall_20, ifa_mean, ifa_median)
