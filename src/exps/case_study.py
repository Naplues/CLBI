# -*- coding:utf-8 -*-
from statistics import mean, quantiles, median

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.helper import *
import matplotlib.pyplot as plt


# case study
def get_num(text):
    key = ['if', 'else', 'switch', 'case', 'for', 'while', 'do', 'break', 'continue', 'return']
    num = 0
    for t in text:
        for k in key:
            if t.find(k) >= 0:
                num += 1
                break
    return num


def call_number(statement):
    statement = statement.strip('\"')
    score = 0
    for char in statement:
        if char == '(':
            score += 1
    return score


def case_study1_CE_2():
    """
    有控制行中有多少比例的代码行包含buggy
    :return:
    """
    ratio_in_buggy_lines, ratio_in_clean_lines, ratio_in_all_lines = [], [], []

    release = 'amq-5.0.0'
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    file_buggy_lines = read_line_level_dataset(release)

    for i in range(len(src_files)):
        file = src_files[i]
        if file not in file_buggy_lines.keys():
            continue

        # Divide all lines into buggy lines and clean lines.
        all_lines, buggy_lines, clean_lines = texts_lines[i], [], []
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        for index in range(len(all_lines)):
            buggy_lines.append(all_lines[index]) if index in buggy_index else clean_lines.append(all_lines[index])

        # CE
        all_num = get_num(all_lines)  # 所有的控制行
        buggy_num = get_num(buggy_lines)  # 有bug的控制行
        clean_num = get_num(clean_lines)  # 无bug的控制行

        if len(buggy_lines) == 0:
            continue
        ratio_in_buggy_lines.append(buggy_num / all_num)
        ratio_in_clean_lines.append(clean_num / all_num)

    print(len(ratio_in_buggy_lines))
    print(len(texts_lines))
    plt.boxplot([ratio_in_buggy_lines])
    plt.savefig('../../result/CaseStudy/CE_2.png')

    data = {'buggy': ratio_in_buggy_lines, 'clean': ratio_in_clean_lines}
    result = DataFrame(data, columns=['buggy'])
    result.to_csv(f'../../result/CaseStudy/CE_2.csv', index=False)


def case_study1_CE_1():
    """
    Buggy/Clean lines 中分别包含多少比例的控制行
    :return:
    """
    ratio_in_buggy_lines, ratio_in_clean_lines, ratio_in_all_lines = [], [], []

    release = 'amq-5.0.0'
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    file_buggy_lines = read_line_level_dataset(release)

    for i in range(len(src_files)):
        file = src_files[i]
        if file not in file_buggy_lines.keys():
            continue

        # Divide all lines into buggy lines and clean lines.
        all_lines, buggy_lines, clean_lines = texts_lines[i], [], []
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        for index in range(len(all_lines)):
            buggy_lines.append(all_lines[index]) if index in buggy_index else clean_lines.append(all_lines[index])

        # CE
        buggy_num = get_num(buggy_lines)
        clean_num = get_num(clean_lines)

        if len(buggy_lines) == 0:
            continue
        ratio_in_buggy_lines.append(buggy_num / len(buggy_lines))
        ratio_in_clean_lines.append(clean_num / len(clean_lines))
        # ratio_in_all_lines.append(all_num / len(all_lines))

    print(len(ratio_in_buggy_lines))
    print(len(texts_lines))
    # plt.boxplot([ratio_in_buggy_lines, ratio_in_clean_lines, ratio_in_all_lines])
    plt.boxplot([ratio_in_buggy_lines, ratio_in_clean_lines])
    plt.savefig('../../result/CaseStudy/CE_1.png')

    data = {'buggy': ratio_in_buggy_lines, 'clean': ratio_in_clean_lines}
    result = DataFrame(data, columns=['buggy', 'clean'])
    result.to_csv(f'../../result/CaseStudy/CE_1.csv', index=False)


def case_study2_NT():
    vector = CountVectorizer(lowercase=False, min_df=2)
    tokenizer = vector.build_tokenizer()
    value_of_buggy, value_of_clean, value_of_all = [], [], []

    release = 'activemq-5.0.0'
    # Loading file and line level dataset
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    file_buggy_lines = read_line_level_dataset(release)

    for i in range(len(src_files)):
        file = src_files[i]
        if file not in file_buggy_lines.keys():
            continue

        # Divide all lines into buggy lines and clean lines.
        all_lines, buggy_lines, clean_lines = texts_lines[i], [], []
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        for index in range(len(all_lines)):
            buggy_lines.append(all_lines[index]) if index in buggy_index else clean_lines.append(all_lines[index])

        t_buggy, t_clean, t_all = [], [], []
        for index in range(len(all_lines)):
            l = len(tokenizer(all_lines[index]))  # NT
            t_buggy.append(l) if index in buggy_index else t_clean.append(l)
            t_all = t_buggy + t_clean

        if len(t_buggy) == 0:
            continue
        value_of_buggy.append(mean(t_buggy))
        value_of_clean.append(mean(t_clean))
        value_of_all.append(mean(t_all))

    print(len(value_of_buggy))
    plt.boxplot([value_of_buggy, value_of_clean, value_of_all])
    plt.savefig('../../result/CaseStudy/NT.png')

    data = {'buggy': value_of_buggy, 'clean': value_of_clean, 'all': value_of_all}
    result = DataFrame(data, columns=['buggy', 'clean', 'all'])
    result.to_csv(f'../../result/CaseStudy/NT.csv', index=False)


def case_study2_NT_2():
    vector = CountVectorizer(lowercase=False, min_df=2)
    tokenizer = vector.build_tokenizer()
    value_of_buggy, value_of_clean, value_of_all = [], [], []

    release = 'amq-5.0.0'
    # Loading file and line level dataset
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    file_buggy_lines = read_line_level_dataset(release)

    for i in range(len(src_files)):
        file = src_files[i]
        if file not in file_buggy_lines.keys():
            continue

        # Divide all lines into buggy lines and clean lines.
        all_lines, buggy_lines, clean_lines = texts_lines[i], [], []
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        for index in range(len(all_lines)):
            buggy_lines.append(all_lines[index]) if index in buggy_index else clean_lines.append(all_lines[index])

        all_num = 0  # 所有的复杂行
        buggy_num = 0
        for index in range(len(all_lines)):
            l = len(tokenizer(all_lines[index]))  # NT
            if l >= 3:
                all_num += 1
                if index in buggy_index:
                    buggy_num += 1

        value_of_buggy.append(buggy_num / all_num)

    print(len(value_of_buggy))
    plt.boxplot([value_of_buggy])
    plt.savefig('../../result/CaseStudy/NT_2.png')

    data = {'buggy': value_of_buggy}
    result = DataFrame(data, columns=['buggy'])
    result.to_csv(f'../../result/CaseStudy/NT_2.csv', index=False)


def case_study2_NFC():
    vector = CountVectorizer(lowercase=False, min_df=2)
    tokenizer = vector.build_tokenizer()
    value_of_buggy, value_of_clean, value_of_all = [], [], []

    release = 'activemq-5.0.0'
    # Loading file and line level dataset
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    file_buggy_lines = read_line_level_dataset(release)

    for i in range(len(src_files)):
        file = src_files[i]
        if file not in file_buggy_lines.keys():
            continue

        # Divide all lines into buggy lines and clean lines.
        all_lines, buggy_lines, clean_lines = texts_lines[i], [], []
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        for index in range(len(all_lines)):
            buggy_lines.append(all_lines[index]) if index in buggy_index else clean_lines.append(all_lines[index])

        t_buggy, t_clean, t_all = [], [], []
        for index in range(len(all_lines)):
            l = call_number(all_lines[index])  # NFC
            t_buggy.append(l) if index in buggy_index else t_clean.append(l)
            t_all = t_buggy + t_clean

        if len(t_buggy) == 0:
            continue
        value_of_buggy.append(mean(t_buggy))
        value_of_clean.append(mean(t_clean))
        value_of_all.append(mean(t_all))

    print(len(value_of_buggy))
    plt.boxplot([value_of_buggy, value_of_clean, value_of_all])
    plt.savefig('../../result/CaseStudy/NFC.png')

    data = {'buggy': value_of_buggy, 'clean': value_of_clean, 'all': value_of_all}
    result = DataFrame(data, columns=['buggy', 'clean', 'all'])
    result.to_csv(f'../../result/CaseStudy/NFC.csv', index=False)


def case_study2_NFC_2():
    value_of_buggy, value_of_clean, value_of_all = [], [], []

    release = 'amq-5.0.0'
    # Loading file and line level dataset
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(release)
    file_buggy_lines = read_line_level_dataset(release)

    for i in range(len(src_files)):
        file = src_files[i]
        if file not in file_buggy_lines.keys():
            continue

        # Divide all lines into buggy lines and clean lines.
        all_lines, buggy_lines, clean_lines = texts_lines[i], [], []
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        for index in range(len(all_lines)):
            buggy_lines.append(all_lines[index]) if index in buggy_index else clean_lines.append(all_lines[index])

        all_num = 0  # 所有的复杂行
        buggy_num = 0
        for index in range(len(all_lines)):
            l = call_number(all_lines[index])  # NFC
            if l >= 1:
                all_num += 1
                if index in buggy_index:
                    buggy_num += 1

        value_of_buggy.append(buggy_num / all_num)

    print(len(value_of_buggy))
    plt.boxplot([value_of_buggy])
    plt.savefig('../../result/CaseStudy/NFC_2.png')

    data = {'buggy': value_of_buggy}
    result = DataFrame(data, columns=['buggy'])
    result.to_csv(f'../../result/CaseStudy/NFC_2.csv', index=False)


def diff_classification():
    releases = PROJECT_RELEASE_LIST
    for release in releases:
        try:
            access_path = result_path + 'Diff_Classification/AccessModel/' + release + '.pk'
            linedp_path = result_path + 'Diff_Classification/LineDPModel/' + release + '.pk'
            access, line_dp = [], []
            with open(access_path, 'rb') as file:
                access = pickle.load(file)
            with open(linedp_path, 'rb') as file:
                line_dp = pickle.load(file)

            r1, r2, r3 = 0, 0, 0
            for e in access:
                if e not in line_dp:
                    r1 += 1
                else:
                    r2 += 1
            for e in line_dp:
                if e not in access:
                    r3 += 1

            print(f'{release}, {r3}, {r2}, {r1}')

        except IOError:
            pass
            # print(f'Error! Not found result file {release}')


def diff_ranking():
    releases = PROJECT_RELEASE_LIST
    for release in releases:
        try:
            access_path = result_path + 'Diff_Ranking/AccessModel/' + release + '.pk'
            linedp_path = result_path + 'Diff_Ranking/LineDPModel/' + release + '.pk'
            access, line_dp = [], []
            with open(access_path, 'rb') as file:
                access = pickle.load(file)
            with open(linedp_path, 'rb') as file:
                line_dp = pickle.load(file)

            increase, decrease = [], []
            for file_line in line_dp.keys():
                rank_of_line_dp = line_dp[file_line]
                rank_of_access = access[file_line]
                # increase
                if rank_of_access < rank_of_line_dp:
                    increase.append(rank_of_line_dp - rank_of_access)
                # decrease
                if rank_of_access > rank_of_line_dp:
                    decrease.append(rank_of_access - rank_of_line_dp)
            print(f'{release}, {decrease}')

        except IOError:
            pass
            # print(f'Error! Not found result file {release}')


if __name__ == '__main__':
    # diff_classification()
    # diff_ranking()
    # case_study1_CE_1()
    # case_study1_CE_2()
    # case_study2_NFC_2()
    case_study2_NT_2()

    pass
