# -*- coding: utf-8 -*-

import sys
from statistics import median, mean

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')

import pandas as pd
from pandas import DataFrame
from src.models.glance import *


def run_Glance(prediction_model, line_level_threshold, effort_aware):
    for project, releases in get_project_releases_dict().items():
        print(f'========== {prediction_model.model_name} CR PREDICTION for {project} =================='[:60])
        for i in range(len(releases) - 1):
            # 1. Loading data. train data index = i, test data index = i + 1
            model = prediction_model(releases[i], releases[i + 1],
                                     line_level_threshold=line_level_threshold,
                                     effort_aware=effort_aware,
                                     test=True)

            model.file_level_prediction()
            model.analyze_file_level_result()

            model.line_level_prediction()
            model.analyze_line_level_result()


def search_parameter(effort_aware=True):
    line_level_thresholds = [.05, .10, .15, .20, .25, .30, .35, .40, ]

    for threshold in line_level_thresholds[::-1]:
        run_Glance(Glance, threshold, effort_aware)


def test_parameter(effort_aware=True):
    line_level_thresholds = [.05, .10, .15, .20, .25, .30, .35, .40, ]

    data = dict()
    names = list()

    mean_list = list()
    for threshold in line_level_thresholds[::-1]:
        m = Glance(line_level_threshold=threshold, effort_aware=effort_aware, test=True)
        df = pd.read_csv(m.line_level_evaluation_file)
        data[m.model_name] = list(df['d2h'])
        names.append(m.model_name)
        mean_list.append(mean(data[m.model_name]))
    print(mean_list)

    result = DataFrame(data, columns=names)

    if effort_aware:
        result.to_csv(f'../../result/RQ1/RQ1-D2H-EA.csv', index=False)
    else:
        result.to_csv(f'../../result/RQ1/RQ1-D2H-MD.csv', index=False)


if __name__ == '__main__':
    #
    # search_parameter(True)
    # test_parameter(True)

    search_parameter(False)
    test_parameter(False)

    pass
