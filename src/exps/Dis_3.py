# -*- coding: utf-8 -*-
from statistics import mean, median

from pandas import DataFrame
import sys

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')
from src.models.explain import *
from src.models.natural import *
from src.models.tools import *
from src.models.glance import *
from src.utils.helper import *


def select_models():
    """
    Select proper models according to different purposes.
    :param exp: Experiment name
    :return: A list of model instances.
    """
    return [NGram(), NGram_C(), PMD(), CheckStyle(), Glance_LR()]


def collect_line_level_summary_result(eva_method=None):
    if eva_method is None:
        eva_method = [mean, median]
    text = ''
    models = select_models()
    for method in eva_method:
        text += f'Approach,Recall,FAR,CE,D2H,MCC,IFA,Recall@20%,ratio\n'
        for model in models:
            df = pd.read_csv(model.line_level_evaluation_file)

            recall = round(method(list(df['recall'])), 3)
            far = round(method(list(df['far'])), 3)
            ce = round(method(list(df['ce'])), 3)
            d2h = round(method(list(df['d2h'])), 3)
            mcc = round(method(list(df['mcc'])), 3)
            ifa = int(method(list(df['ifa'])))
            recall_20 = round(method(list(df['recall_20'])), 3)
            ratio = round(method(list(df['ratio'])), 3)
            text += f'{model.model_name},{recall},{far},{ce},{d2h},{mcc},{ifa},{recall_20},{ratio}\n'
        text += '\n'
    save_csv_result(f'../../result/Dis_3/', f'Performance_Summary.csv', text)


# =================== Line level result in terms of different Performance Indicators experiments ================
def collect_line_level_by_indicators():
    models = select_models()
    indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']
    for indicator in indicators:
        data = dict()
        for model in models:
            data[model.model_name] = pd.read_csv(model.line_level_evaluation_file)[indicator].tolist()

        ratio = DataFrame(data, columns=[model.model_name for model in models])
        ratio.to_csv(f'../../result/Dis_3/Performance Indicators/{indicator}.csv', index=False)


# =================== Line level result in terms of different Indicators and project experiments ================
def collect_line_level_by_indicators_project():
    indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']
    for indicator in indicators:

        all_data = read_data_from_file(f'../../result/Dis_3/Performance Indicators/{indicator}.csv')[1:]

        text = 'NGram,NGram-C,PMD,CheckStyle,Glance-LR\n'
        count = 0
        for project, releases in get_project_releases_dict().items():
            m0, m1, m2, m3, m4, = [], [], [], [], []
            for i in range(len(releases) - 1):
                values = all_data[count].split(',')
                m0.append(float(values[0]))
                m1.append(float(values[1]))
                m2.append(float(values[2]))
                m3.append(float(values[3]))
                m4.append(float(values[4]))

                count += 1
            text += f'{project},{mean(m0)},{mean(m1)},{mean(m2)},{mean(m3)},{mean(m4)}\n'

        save_csv_result(f'../../result/Dis_3/Performance Indicators/', f'avg-{indicator}.csv', text)


if __name__ == '__main__':
    #
    collect_line_level_summary_result()
    collect_line_level_by_indicators()
    collect_line_level_by_indicators_project()
    pass
