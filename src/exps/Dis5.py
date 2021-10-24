# -*- coding:utf-8 -*-

from src.utils.helper import *


def fun_project(proj, releases):
    debt_bug, no_debt_bug, debt_no_bug, no_debt_no_bug = 0, 0, 0, 0
    for release in releases[1:]:
        path = f'{root_path}temp/{release}.csv'
        r1, r2, r3, r4 = fun_release(release, path)
        debt_bug += r1
        no_debt_bug += r2
        debt_no_bug += r3
        no_debt_no_bug += r4
    print(f'{proj},{debt_bug},{no_debt_bug},{debt_no_bug},{no_debt_no_bug}')


def fun_release(release, file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    debt, bug, debt_bug, debt_no_bug, no_debt_bug, no_debt_no_bug = 0, 0, 0, 0, 0, 0
    for line in data[1:]:
        ss = line.strip().split(',')
        if ss[1] >= '1':
            debt += 1
            if ss[2] == '1':
                bug += 1
                debt_bug += 1
            else:
                debt_no_bug += 1
        else:
            if ss[2] == '1':
                bug += 1
                no_debt_bug += 1
            else:
                no_debt_no_bug += 1

    # print(f'{release},{debt},{bug},{debt_bug},{no_debt_bug},{debt_no_bug},{no_debt_no_bug}')
    return debt_bug, no_debt_bug, debt_no_bug, no_debt_no_bug


if __name__ == '__main__':

    for project, releases in get_project_releases_dict().items():
        fun_project(project, releases)
