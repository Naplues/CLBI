# -*- coding: utf-8 -*-
from statistics import mean, median

from pandas import DataFrame

from src.models.explain import *
from src.models.natural import *
from src.models.tools import *
from src.models.glance import *
from src.utils.helper import *


def select_models(exp: str = 'RQ2'):
    """
    Select proper models according to different purposes.
    :param exp: Experiment name
    :return: A list of model instances.
    """
    if exp == 'RQ2':
        m = [NGram(), NGram_C(), TMI_LR(), TMI_MNB(), TMI_SVM(), TMI_DT(), TMI_RF(), LineDP(), Glance_MD(), Glance_EA()]
    elif exp == 'RQ3':
        m = [PMD(), CheckStyle(), Glance_MD(), Glance_EA()]
    else:
        m = []
    return m


def collect_line_level_summary_result(exp: str = 'RQ2', eva_method=None):
    if eva_method is None:
        eva_method = [mean, median]
    text = ''
    models = select_models(exp)
    for method in eva_method:
        text += f'Approach,Recall,FAR,CE,D2H,MCC,IFA,Recall@20%,ER,RI\n'
        for model in models:
            df = pd.read_csv(model.line_level_evaluation_file)

            recall = round(method(list(df['recall'])), 3)
            far = round(method(list(df['far'])), 3)
            ce = round(method(list(df['ce'])), 3)
            d2h = round(method(list(df['d2h'])), 3)
            mcc = round(method(list(df['mcc'])), 3)
            ifa = int(method(list(df['ifa'])))
            recall_20 = round(method(list(df['recall_20'])), 3)
            ER = round(method(list(df['ER'])), 3)
            RI = round(method(list(df['RI'])), 3)
            text += f'{model.model_name},{recall},{far},{ce},{d2h},{mcc},{ifa},{recall_20},{ER},{RI}\n'
        text += '\n'
    save_csv_result(f'../../result/{exp}/', f'{exp}-line_level_result.csv', text)


# ======================== Line level result in terms of different indicators experiments =============================
def collect_line_level_by_indicators(exp, indicator=None):
    models = select_models(exp)
    indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']
    for indicator in indicators:
        data = dict()
        for model in models:
            data[model.model_name] = pd.read_csv(model.line_level_evaluation_file)[indicator].tolist()

        ratio = DataFrame(data, columns=[model.model_name for model in models])
        ratio.to_csv(f'../../result/{exp}/{exp}-{indicator}.csv', index=False)


if __name__ == '__main__':
    #
    experiments = ["RQ2", "RQ3"]
    for experiment in experiments:
        # collect_line_level_summary_result(experiment)
        collect_line_level_by_indicators(experiment)

        pass
    pass
