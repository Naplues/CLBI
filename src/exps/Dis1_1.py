# -*- coding:utf-8 -*-

import sys

import warnings
import pandas as pd
from pandas import DataFrame

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')
from src.models.glance import *
from statistics import *

output_path = '../../result/Dis1_1/'


def get_effort_of_Glance(clf):
    print(clf)
    total_data = read_data_from_file(f'{output_path}effort-total.csv')[1:]
    text = 'release,review_effort,total_effort,ratio\n'
    count = 0
    for project, releases in get_project_releases_dict().items():
        for i in range(len(releases) - 1):
            total_effort = int(total_data[count].split(',')[1])
            file_path = f'{root_path}Result/{clf}/line_result/{project}/{releases[i + 1]}-result.csv'
            review_effort = len(read_data_from_file(file_path)) - 1
            text += f'{releases[i + 1]},{review_effort},{total_effort},{round(review_effort / total_effort, 3)}\n'
            count += 1
    save_csv_result(output_path, f'effort-{clf}.csv', text)


def summary():
    MD = read_data_from_file(f'{output_path}effort-Glance-MD.csv')[1:]
    EA = read_data_from_file(f'{output_path}effort-Glance-EA.csv')[1:]
    LR = read_data_from_file(f'{output_path}effort-Glance-LR.csv')[1:]

    text = 'release,MD_effort,EA_effort,LR_effort\n'
    count = 0
    for project, releases in get_project_releases_dict().items():
        MD_data, EA_data, LR_data = [], [], []
        for i in range(len(releases) - 1):
            MD_data.append(float(MD[count].split(',')[3]))
            EA_data.append(float(EA[count].split(',')[3]))
            LR_data.append(float(LR[count].split(',')[3]))
            count += 1
        text += f'{project},{median(MD_data)},{median(EA_data)},{median(LR_data)}\n'

    save_csv_result(output_path, f'effort-comparison.csv', text)


if __name__ == '__main__':
    for clf in ['Glance-MD', 'Glance-EA', 'Glance-LR']:
        # get_effort_of_Glance(clf)
        pass
    summary()
    pass
