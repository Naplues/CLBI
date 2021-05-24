# -*- coding:utf-8 -*-

from src.utils.helper import *
from numpy import *


def diff_classification():
    access_model = 'AccessModel'
    target_model = 'PMDModel'  # PMDModel CheckStyleModel
    print('Release,Access,Target,Hit,Over')
    for project, releases in get_project_releases_dict().items():
        print(project, end=',')

        A, T, A_T, A__T = 0, 0, 0, 0
        for release in releases[1:]:
            data_access = set(load_pk_result(f'{result_path}Diff_Classification/{access_model}/{release}.pk'))
            data_target = set(load_pk_result(f'{result_path}Diff_Classification/{target_model}/{release}.pk'))
            a, t, = len(data_access), len(data_target)
            a_t, a__t = len(data_access & data_target), len(data_access - data_target)
            A += a
            T += t
            A_T += a_t
            A__T += a__t
        A /= len(releases) - 1
        T /= len(releases) - 1
        A_T /= len(releases) - 1
        A__T /= len(releases) - 1
        if T == 0:
            print(f'{int(A)},{int(T)},-,-')
        else:
            hit, over = A_T / T, A__T / T
            print(f'{int(A)},{int(T)},{"%.3f" % hit},{"%.3f" % over}')


def diff_ranking():
    access_model = 'AccessModel'
    target_model = 'CheckStyleModel'  # PMDModel CheckStyleModel
    text = ''
    for project, releases in get_project_releases_dict().items():

        print(project, end=',')
        increase, decrease = [], []
        for release in releases[1:]:
            data_access = load_pk_result(f'{result_path}Diff_Ranking/{access_model}/{release}.pk')
            data_target = load_pk_result(f'{result_path}Diff_Ranking/{target_model}/{release}.pk')
            i, d = [], []
            t, text_lines, label, filename = read_file_level_dataset(release)
            len_dict = {}
            for index in range(len(filename)):
                len_dict[filename[index]] = len(text_lines[index])

            for file_line in data_target.keys():
                if file_line not in data_access.keys():
                    continue
                rank_of_access = data_access[file_line]
                rank_of_target = data_target[file_line]
                # increase
                if rank_of_access < rank_of_target:
                    i.append((rank_of_target - rank_of_access) / len_dict[file_line.split(':')[0]])
                # decrease
                if rank_of_access > rank_of_target:
                    d.append((rank_of_access - rank_of_target) / len_dict[file_line.split(':')[0]])

            increase.append(mean(i))
            decrease.append(mean(d))

        text += project + ',' + str(increase).replace('[', '').replace(']', '') + '\n'
        text += project + ',' + str(decrease).replace('[', '').replace(']', '') + '\n'
    print(text)
    save_result('C:\\Users\\GZQ\\Desktop\\d.csv', text)


def calc_ranking_average_value():
    tool = 'pmd'  # checkstyle
    with open(f'C:\\Users\\GZQ\\Desktop\\{tool}.csv', 'r') as file:
        data = file.readlines()
    flag = 0
    for line in data:
        if flag == 0:
            ss = line.strip().split(',')

        flag = 1 - flag


if __name__ == '__main__':
    # diff_ranking()
    calc_ranking_average_value()
