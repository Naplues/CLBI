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


if __name__ == '__main__':
    r1, r2 = case_study2()
    print('\n'.join(r1))
    print('=' * 80)
    print('\n'.join(r2))
