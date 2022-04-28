# -*- coding: utf-8 -*-


import sys

import warnings
import pandas as pd
from pandas import DataFrame

sys.path.append('C:/Users/gzq-712/Desktop/Git/CLDP/')
from src.models.glance import *
from statistics import *

output_path = '../../result/Dis_4/'


def average(app):
    all_data = read_data_from_file(f'{output_path}time-{app}.csv')[1:]

    text = 'project,build_avg_time,pred_avg_time\n'
    count = 0
    for project, releases in get_project_releases_dict().items():
        build_list, pred_list = [], []
        for i in range(len(releases) - 1):
            time = all_data[count].split(',')
            build_list.append(float(time[1]))
            pred_list.append(float(time[2]))
            count += 1
        text += f'{project},{mean(build_list)},{mean(pred_list)}\n'

    save_csv_result(output_path, f'avg-time-{app}.csv', text)


if __name__ == '__main__':
    for approach in ['NGram', 'LineDP', 'Glance-MD', 'Glance-EA', 'Glance-LR']:
        average(approach)
    pass
