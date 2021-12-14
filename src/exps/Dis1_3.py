# -*- coding: utf-8 -*-

import sys

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')
from statistics import mean, median
import pandas as pd
from pandas import DataFrame

from src.models.glance import *

output_path = '../../result/Dis1_3/'


def select_models():
    """
    Select proper models according to different purposes.
    :param exp: Experiment name
    :return: A list of model instances.
    """
    # , Glance_B_NT_noCC(), Glance_B_NFC_noCC
    return [Glance_LR(), Glance_B_noCC(), Glance_B_NT(), Glance_B_NT_noCC(), Glance_B_NFC(), Glance_B_NFC_noCC()]


def collect_line_level_summary_result(eva_method=None):
    if eva_method is None:
        eva_method = [mean, median]
    text = ''
    for method in eva_method:
        text += f'Approach,Recall,FAR,CE,D2H,MCC,IFA,Recall@20%,ratio\n'
        for model in select_models():
            df = pd.read_csv(model.line_level_evaluation_file)

            recall = round(method(list(df['recall'])), 3)
            far = round(method(list(df['far'])), 3)
            ce = round(method(list(df['ce'])), 3)
            d2h = round(method(list(df['d2h'])), 3)
            mcc = round(method(list(df['mcc'])), 3)
            ifa = int(method(list(df['ifa'])))
            recall_20 = round(method(list(df['recall_20'])), 3)
            ratio = round(method(list(df['ratio'])), 3)
            # ER = round(method(list(df['ER'])), 3)
            # RI = round(method(list(df['RI'])), 3)
            text += f'{model.model_name},{recall},{far},{ce},{d2h},{mcc},{ifa},{recall_20},{ratio}\n'
        text += '\n'
    save_csv_result(output_path, f'Performance_Summary.csv', text)


# =================== Line level result in terms of different Performance Indicators experiments ================
def collect_line_level_by_indicators():
    models = select_models()
    indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']
    for indicator in indicators:
        data = dict()
        for model in models:
            data[model.model_name] = pd.read_csv(model.line_level_evaluation_file)[indicator].tolist()

        ratio = DataFrame(data, columns=[model.model_name for model in models])
        ratio.to_csv(f'{output_path}Performance Indicators/{indicator}.csv', index=False)


if __name__ == '__main__':
    collect_line_level_summary_result()
    collect_line_level_by_indicators()
    pass
