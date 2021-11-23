# -*- coding: utf-8 -*-

import sys

import warnings
import pandas as pd
from pandas import DataFrame

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')
from src.models.glance import *
from statistics import *

# Ignore warning information
warnings.filterwarnings('ignore')

line_level_thresholds = [.05, .10, .15, .20, .25, .30, .35, .40, .45, .50]
indicators = ['recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio']

output_path = '../../result/RQ1/'
make_path(output_path)


def select_model(file_level_classifier, line_level_threshold, train='', test=''):
    if file_level_classifier == 'MD':
        model = Glance_MD(train, test, line_threshold=line_level_threshold, test=True)
    elif file_level_classifier == 'EA':
        model = Glance_EA(train, test, line_threshold=line_level_threshold, test=True)
    else:
        model = Glance_LR(train, test, line_threshold=line_level_threshold, test=True)
    return model


def search_parameter_Glance(clf):
    for threshold in line_level_thresholds:
        for project, releases in get_project_releases_dict().items():
            for i in range(len(releases) - 1):
                # 1. Loading data. train data index = i, test data index = i + 1
                model = select_model(clf, threshold, releases[i], releases[i + 1])

                print(f'========== {model.model_name} CR PREDICTION for {releases[i + 1]} =================='[:60])
                model.file_level_prediction()
                model.analyze_file_level_result()

                model.line_level_prediction()
                model.analyze_line_level_result()


def test_parameter(clf):
    print(f'Glance {clf}')
    # 水平展示的变化数据, 列名为与之
    summary_data_horizontal, summary_data_vertical = list(), dict()
    for indicator in indicators:
        detail_data, column_names, mean_list = dict(), list(), list()
        for threshold in line_level_thresholds:
            model = select_model(clf, threshold)
            column_names.append(model.model_name)
            detail_data[model.model_name] = list(pd.read_csv(model.line_level_evaluation_file)[indicator])

            mean_list.append(round(mean(detail_data[model.model_name]), 3))

        summary_data_horizontal.append(mean_list)
        summary_data_vertical[indicator] = mean_list

        detail_result = DataFrame(detail_data, index=get_test_releases_list(), columns=column_names)

        make_path(f'{output_path}RQ1-Glance-{clf}/')
        detail_result.to_csv(f'{output_path}RQ1-Glance-{clf}/{indicator}.csv', index=True)

    threshold_indices = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%', ]
    summary_result = DataFrame(summary_data_horizontal, index=indicators, columns=threshold_indices)
    summary_result.to_csv(f'{output_path}RQ1-summary-Glance-{clf}-horizontal.csv', index=True)
    summary_result = DataFrame(summary_data_vertical, index=threshold_indices, columns=indicators)
    summary_result.to_csv(f'{output_path}RQ1-summary-Glance-{clf}-vertical.csv', index=True)


if __name__ == '__main__':
    #
    file_level_classifiers = ['MD', 'EA', 'LR']
    for classifier in file_level_classifiers:
        # search_parameter_Glance(classifier)
        test_parameter(classifier)

    pass
