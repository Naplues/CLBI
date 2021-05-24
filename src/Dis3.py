# -*- coding:utf-8 -*-

from src.utils.helper import *


def TP_FP(cp_result_path):
    result = 'project,TP,FP\n'
    for project in projects:
        tp, fp, tn, fn, p, r, f1, mcc = 0, 0, 0, 0, .0, .0, .0, .0
        data = read_data_from_file(f'{cp_result_path}/file_level_evaluation_{project}.csv')
        l = len(data[1:])
        for item in data[1:]:
            ss = item.split(',')
            tp += int(ss[1])
            fp += int(ss[2])
            tn += int(ss[3])
            fn += int(ss[4])
            p += float(ss[5])
            r += float(ss[6])
            f1 += float(ss[7])
            mcc += float(ss[8])
        result += f'{project},{int(tp / l)},{int(fp / l)},{int(tn / l)},{int(fn / l)},{p / l},{r / l},{f1 / l},{mcc / l}\n'

    print(result)
    save_csv_result(cp_result_path, f'file_level_evaluation.csv', result)


if __name__ == '__main__':
    TP_FP(f'{root_path}Result/CP/TMI_SVM_Model_50/')
