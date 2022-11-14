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

exp_result_root_path = f'../../result/Dis_5/'


def select_models():
    return [NGram(), NGram_C(), TMI_LR(), TMI_MNB(), TMI_SVM(), TMI_DT(), TMI_RF(), LineDP(), PMD(), CheckStyle(),
            Glance_MD(), Glance_EA(), Glance_LR()]


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
    save_csv_result(exp_result_root_path, f'Performance_Summary.csv', text)


# =================== Line level result in terms of different Performance Indicators experiments ================
def collect_line_level_by_indicators():
    models = select_models()
    indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']
    for indicator in indicators:
        data = dict()
        for model in models:
            data[model.model_name] = pd.read_csv(model.line_level_evaluation_file)[indicator].tolist()

        ratio = DataFrame(data, columns=[model.model_name for model in models])

        make_path(f'{exp_result_root_path}Performance Indicators/')
        ratio.to_csv(f'{exp_result_root_path}Performance Indicators/{indicator}.csv', index=False)


# =================== Line level result in terms of different Indicators and project experiments ================
def collect_line_level_by_indicators_project():
    indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']
    for indicator in indicators:

        all_data = read_data_from_file(f'{exp_result_root_path}Performance Indicators/{indicator}.csv')[1:]

        text = 'NGram,NGram-C,TMI-LR,TMI-MNB,TMI-SVM,TMI-DT,TMI-RF,LineDP,PMD,CheckStyle,Glance-MD,Glance-EA,Glance-LR\n'
        count = 0
        for project, releases in get_project_releases_dict().items():
            m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12 = [], [], [], [], [], [], [], [], [], [], [], [], []
            for i in range(len(releases) - 1):
                values = all_data[count].split(',')
                m0.append(float(values[0]))
                m1.append(float(values[1]))
                m2.append(float(values[2]))
                m3.append(float(values[3]))
                m4.append(float(values[4]))
                m5.append(float(values[5]))
                m6.append(float(values[6]))
                m7.append(float(values[7]))
                m8.append(float(values[8]))
                m9.append(float(values[9]))
                m10.append(float(values[10]))
                m11.append(float(values[11]))
                m12.append(float(values[12]))

                count += 1
            text += f'{project},{mean(m0)},{mean(m1)},{mean(m2)},{mean(m3)},{mean(m4)},{mean(m5)},{mean(m6)},' \
                    f'{mean(m7)},{mean(m8)},{mean(m9)},{mean(m10)},{mean(m11)},{mean(m12)}\n'

        make_path(f'{exp_result_root_path}Performance Indicators/')
        save_csv_result(f'{exp_result_root_path}Performance Indicators/', f'avg-{indicator}.csv', text)


if __name__ == '__main__':
    #
    # collect_line_level_summary_result()
    collect_line_level_by_indicators()
    collect_line_level_by_indicators_project()

    pass
