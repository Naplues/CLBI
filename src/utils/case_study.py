# -*- coding:utf-8 -*-

from src.utils.helper import *


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


def case_study():
    p = 'activemq-5.0.0'
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(p)
    file_buggy_lines = read_line_level_dataset(p)
    buggy_files = file_buggy_lines.keys()
    ratio = []
    ratio2 = []
    for i in range(len(src_files)):
        file = src_files[i]
        if file not in buggy_files:
            continue
        all = texts_lines[i]
        buggy = []
        for index in file_buggy_lines[file]:
            buggy.append(all[index - 1])
        all_num = get_num(all)
        buggy_num = get_num(buggy)
        if all_num != 0:
            ratio.append(str(buggy_num / len(buggy)))
            ratio2.append(str(all_num / len(all)))

    return ratio, ratio2


def call_number(statement):
    statement = statement.strip('\"')
    score = 0
    for char in statement:
        if char == '(':
            score += 1
    return score


def case_study2():
    p = 'activemq-5.0.0'
    texts, texts_lines, numeric_labels, src_files = read_file_level_dataset(p)
    file_buggy_lines = read_line_level_dataset(p)
    buggy_files = file_buggy_lines.keys()
    ratio_buggy = []
    ratio_clean = []
    vector = CountVectorizer(lowercase=False, min_df=2)
    tokenizer = vector.build_tokenizer()
    for i in range(len(src_files)):
        file = src_files[i]
        if file not in buggy_files:
            continue
        all = texts_lines[i]
        buggy_index = [e - 1 for e in file_buggy_lines[file]]
        t_b = []
        t_c = []
        for index in range(len(all)):
            # l = call_number(all[index])
            l = len(tokenizer(all[index]))
            if index in buggy_index:
                t_b.append(l)
            else:
                t_c.append(l)
        ratio_buggy.append(str(np.mean(t_b)))
        ratio_clean.append(str(np.mean(t_c)))
    return ratio_buggy, ratio_clean


def study():
    r1, r2 = case_study2()
    print('\n'.join(r1))
    print('=' * 80)
    print('\n'.join(r2))


def diff_classification():
    releases = get_project_release_list()
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
    releases = get_project_release_list()
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
    diff_ranking()
