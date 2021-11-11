# -*- coding: utf-8 -*-
from statistics import mean, median

from src.models.explain import *
from src.models.glance import Glance_EA, Glance_MD
from src.models.natural import NGram, NGram_C
from src.models.tools import PMD, CheckStyle
from src.utils.helper import *


class RQ(object):
    def __init__(self, model: BaseModel, project: str, release: str):
        self.model = model
        self.project_name = project
        self.test_release = release

        self.result_path = f'{root_path}Result/{self.model.model_name}/'
        self.line_level_result_path = f'{self.result_path}line_result/'
        self.line_level_result_file = f'{self.line_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        df = pd.read_csv(self.line_level_result_file)
        self.predicted_buggy_lines = list(df['predicted_buggy_lines'])


def prediction_result(prediction_model):
    text = f'      \t========== {prediction_model.model_name} ==========\n'
    text += '      \tRecall\tFAR  \tCE   \tD2H  \tMCC  \tIFA \tRecall@20%\tER\tRI\n'

    df = pd.read_csv(prediction_model.line_level_evaluation_file)

    recall = round(mean(list(df['recall'])), 3)
    far = round(mean(list(df['far'])), 3)
    ce = round(mean(list(df['ce'])), 3)
    d2h = round(mean(list(df['d2h'])), 3)
    mcc = round(mean(list(df['mcc'])), 3)
    ifa = int(mean(list(df['ifa'])))
    recall_20 = round(mean(list(df['recall_20'])), 3)
    ER = round(mean(list(df['ER'])), 3)
    RI = round(mean(list(df['RI'])), 3)
    text += f'Mean  : {recall}\t{far}\t{ce}\t{d2h}\t{mcc}\t{ifa}\t{recall_20}\t{ER}\t{RI}\n'

    recall = round(median(list(df['recall'])), 3)
    far = round(median(list(df['far'])), 3)
    ce = round(median(list(df['ce'])), 3)
    d2h = round(median(list(df['d2h'])), 3)
    mcc = round(median(list(df['mcc'])), 3)
    ifa = int(median(list(df['ifa'])))
    recall_20 = round(median(list(df['recall_20'])), 3)
    ER = round(median(list(df['ER'])), 3)
    RI = round(median(list(df['RI'])), 3)
    text += f'Median: {recall}\t{far}\t{ce}\t{d2h}\t{mcc}\t{ifa}\t{recall_20}\t{ER}\t{RI}\n'

    print(text)
    return text


def show_result():
    text = ''
    text += prediction_result(Glance_MD('', ''))
    text += prediction_result(Glance_EA('', ''))

    text += prediction_result(TMI_LR('', ''))
    text += prediction_result(TMI_SVM('', ''))
    text += prediction_result(TMI_MNB('', ''))
    text += prediction_result(TMI_DT('', ''))
    text += prediction_result(TMI_RF('', ''))
    text += prediction_result(LineDP('', ''))

    text += prediction_result(NGram('', ''))
    text += prediction_result(NGram_C('', ''))

    text += prediction_result(PMD('', ''))
    text += prediction_result(CheckStyle('', ''))

    save_csv_result('../../result/', f'RQ1.csv', text)


# ========================= Run RQ2 experiments =================================
def run(prediction_model):
    print(f'========== {prediction_model.model_name} ================='[:40])
    texts = '\n'
    df = pd.read_csv(prediction_model.line_level_evaluation_file)
    new_file = df['ratio']

    save_csv_result('../../result/', f'RQ2-{prediction_model.model_name}.csv', texts)


if __name__ == '__main__':
    #
    run(Glance_EA('', ''))
    run(Glance_MD('', ''))

    run(TMI_LR('', ''))
    run(TMI_MNB('', ''))
    run(TMI_SVM('', ''))
    run(TMI_DT('', ''))
    run(TMI_RF('', ''))
    run(LineDP('', ''))

    run(NGram('', ''))
    run(NGram_C('', ''))

    run(PMD('', ''))
    run(CheckStyle('', ''))

    # show_result()
    pass
