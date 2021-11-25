# -*- coding: utf-8 -*-

import sys

import warnings
import pandas as pd
from pandas import DataFrame

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')
from src.models.glance import *

# Ignore warning information
warnings.filterwarnings('ignore')

indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']

output_path = '../../result/RQ1/'
make_path(output_path)


def select_model(file_level_classifier, train='', test=''):
    if file_level_classifier == 'MD':
        model = Glance_MD(train, test)
    elif file_level_classifier == 'EA':
        model = Glance_EA(train, test)
    else:
        model = Glance_LR(train, test)
    return model


def test_parameter(clf):
    print(f'======================== Glance {clf} ===========================')
    detail_data, column_names = list(), list()
    model = select_model(clf)
    data = pd.read_csv(model.line_level_evaluation_file)[indicators]
    last = 0
    for project, release in get_project_releases_dict().items():
        start, end = last, last + len(release[1:])
        detail_data.append(list(data.iloc[start:end].mean(axis=0)))
        column_names.append(project)
        last = end

    summary_result = DataFrame(detail_data, index=get_project_list(), columns=indicators)
    summary_result.to_csv(f'{output_path}Glance-{clf}.csv', index=True)


if __name__ == '__main__':
    #
    file_level_classifiers = ['MD', 'EA', 'LR']
    for classifier in file_level_classifiers:
        test_parameter(classifier)
    pass
