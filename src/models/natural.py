# -*- coding:utf-8 -*-

# n-gram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from src.utils.eval import evaluation
from src.utils.helper import *
import math


def get_tokenize(release, train_text, n):
    """
    :param release:
    :param train_text:
    :param n:
    :return:
    """
    for i in range(1, n + 1):
        file_path = f'{result_path}NBF/{release}/{i}-gram.txt'
        if os.path.exists(file_path):
            continue
        ngram_vector = CountVectorizer(lowercase=False, stop_words=None, ngram_range=(i, i))
        corpus = [' '.join(train_text)]
        train_vtr = ngram_vector.fit_transform(corpus)
        text = '\n'.join([f'{token}:{train_vtr[0, index]}' for token, index in ngram_vector.vocabulary_.items()])

        save_result(file_path, text)
        print(f'{file_path} word output finish!')


def get_vocabulary(release):
    return set([line.split(':')[0] for line in read_data_from_file(f'{result_path}NBF/{release}/1-gram.txt')])


def count_sequence(target_release, n):
    """
    Count the frequency of each sequence
    :param target_release:
    :param n:
    :return:
    """
    seq_dict = {}
    lines = read_data_from_file(f'{result_path}NBF/{target_release}/{n}-gram.txt')
    for line in lines:
        split = line.split(':')
        seq, count = split[0], int(split[1])
        seq_dict[seq] = count
    return seq_dict


def build_global_n_gram(target_release, n):
    for i in range(1, n + 1):
        if i == 1:
            # 序列字典
            seq_n_dict = count_sequence(target_release, i)
            # 字典大小
            size_of_v = len(seq_n_dict)
            # 所有序列出现的次数总和
            size_of_all_seq = sum([count for token, count in seq_n_dict.items()])
            # 未出现的词的概率
            n_gram_dict = {'-': 1 / (size_of_all_seq + size_of_v)}
            # 计算每个序列对应的概率
            for token in seq_n_dict.keys():
                n_gram_dict[token] = (seq_n_dict[token] + 1) / (size_of_all_seq + size_of_v)
            # 存储概率字典
            dump_pk_result(f'{result_path}NBF/{target_release}/n-gram.pk', n_gram_dict)
        else:
            n_gram_prefix_dict = {}
            n_gram_suffix_dict = {}
            # sequence with length of n, sequence with length of n-1
            seq_n_dict, seq_n_1_dict = count_sequence(target_release, i), count_sequence(target_release, i - 1)

            for seq, count_seq in seq_n_dict.items():
                # seq: 'java util HashMap' count_seq:37
                split = seq.split(' ')
                # 前缀序列和后缀序列
                seq_prefix, seq_suffix = ' '.join(split[:-1]), ' '.join(split[1:])
                # 前缀对应的尾部单词 后缀对应的首部单词
                token_prefix, token_suffix = split[-1], split[0]
                # 前缀序列的频率 后缀序列的频率 NOTE 前缀或者后缀一定存在
                count_prefix, count_suffix = seq_n_1_dict[seq_prefix], seq_n_1_dict[seq_suffix]
                # 词库的大小
                size_of_v = len(get_vocabulary(target_release))
                # 前缀常量 后缀常量
                prefix_c, suffix_c = 1 / (count_prefix + size_of_v), 1 / (count_suffix + size_of_v)
                # 获取 序列前缀的字典 序列后缀的字典
                d_prefix = n_gram_prefix_dict[seq_prefix] if seq_prefix in n_gram_prefix_dict else {'-': prefix_c}
                d_suffix = n_gram_suffix_dict[seq_suffix] if seq_suffix in n_gram_suffix_dict else {'-': suffix_c}

                d_prefix[token_prefix] = (count_seq + 1) / (count_prefix + size_of_v)
                d_suffix[token_suffix] = (count_seq + 1) / (count_suffix + size_of_v)

                n_gram_prefix_dict[seq_prefix] = d_prefix
                n_gram_suffix_dict[seq_suffix] = d_suffix

            dump_pk_result(f'{result_path}NBF/{target_release}/{i}-gram_prefix.pk', n_gram_prefix_dict)
            dump_pk_result(f'{result_path}NBF/{target_release}/{i}-gram_suffix.pk', n_gram_suffix_dict)
    print(f'==================== {n}-gram dict of {target_release} output finish ========================'[:80])


def build_global_language_model(target_release, order):
    print(f'Building global {order}-gram model for {target_release}')
    # read line level text from the target release
    train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(target_release)
    # tokenize a source file
    get_tokenize(target_release, train_text, order)
    build_global_n_gram(target_release, order)
    print(f'The {order}-gram model has been built finish!')


def predict_global_entropy(test_file, analysis, ngram_dict, prefix_dict, suffix_dict, order):
    entropy_of_each_file = []
    # process each file
    for line_index in range(len(test_file)):
        line = test_file[line_index]
        # process each line
        words_in_line = analysis(line.strip())
        # the number of words in the line
        num_of_words = len(words_in_line)
        # entropy = -1/n * sum( p(ti,h) * log(p(ti,h)) )
        entropy = -1
        if num_of_words > 0:
            ent_of_token = 0
            for i in range(num_of_words):
                if order == 1:
                    # n-gram n==1
                    prob = ngram_dict.get(words_in_line[i], ngram_dict['-'])
                else:
                    # n-gram n>=2
                    # prolog
                    start = 0 if i - (order - 1) < 0 else i - (order - 1)
                    # 前缀 当前词 当前词的概率
                    prefix, current, prob_prefix = ' '.join(words_in_line[start:i]), words_in_line[i], 0
                    if len(prefix) == 0:
                        prob_prefix = ngram_dict.get(current, ngram_dict['-'])
                    else:
                        if prefix not in prefix_dict:
                            prob_prefix = prefix_dict['-']
                        else:
                            prob_prefix = prefix_dict[prefix].get(current, prefix_dict[prefix]['-'])

                    # epilog
                    end = num_of_words if i + (order - 1) > num_of_words else i + (order - 1)
                    suffix, current, prob_suffix = ' '.join(words_in_line[i:end]), words_in_line[i], 0
                    if len(suffix) == 0:
                        prob_suffix = ngram_dict.get(current, ngram_dict['-'])
                    else:
                        if suffix not in suffix_dict:
                            prob_suffix = suffix_dict['-']
                        else:
                            prob_suffix = suffix_dict[suffix].get(current, suffix_dict[suffix]['-'])

                    prob = (prob_prefix + prob_suffix) / 2

                # ent_of_token += prob * math.log(prob)
                ent_of_token += math.log(prob)

            entropy = -(ent_of_token / num_of_words)
            # print(entropy)
        entropy_of_each_file.append(entropy)
    return np.array(entropy_of_each_file)


# OK 进行代码行级别的排序
def Ngram_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, n_gram_order):
    """
    Ranking line-level defect-prone lines using TMI-LR model
    :param proj:
    :param vector:
    :param classifier:
    :param test_text_lines:
    :param test_filename:
    :param test_predictions:
    :param out_file:
    :param n_gram_order
    :return:
    """
    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}
    defect_cf_dict = {}
    effort_cf_dict = {}

    # 预测为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])

    # Language Model 变量
    analysis = CountVectorizer(lowercase=False, stop_words=None).build_analyzer()
    ngram_dict = load_pk_result(f'{result_path}NBF/{proj}/n-gram.pk')
    prefix_dict, suffix_dict = {}, {}
    if n_gram_order > 1:
        prefix_dict = load_pk_result(f'{result_path}NBF/{proj}/{n_gram_order}-gram_prefix.pk')
        suffix_dict = load_pk_result(f'{result_path}NBF/{proj}/{n_gram_order}-gram_suffix.pk')

    # 对预测为有bug的文件逐个进行解释结果来进行代码行级别的预测
    for i in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[i]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的文件被预测为有bug,但实际上没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ####################################### 核心部分 #################################################
        # 统计 每一行中出现 risk tokens 的个数, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数

        hit_count = predict_global_entropy(target_file_lines, analysis, ngram_dict, prefix_dict, suffix_dict,
                                           n_gram_order)

        # ####################################### 核心部分 #################################################

        # 根据命中次数对所有代码行进行降序排序, 按照排序后数值从大到小的顺序显示每个元素在原列表中的索引(i.e., 行号-1)
        # line + 1,因为原列表中代表行号的索引是从0开始计数而不是从1开始
        sorted_index = np.argsort(-hit_count)
        sorted_line_number = [line + 1 for line in sorted_index.tolist()]
        # 原始未经过调整的列表
        ranked_list_dict[target_file_name] = sorted_line_number

        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################
        # 需要调整为最差排序的列表,当分数相同时
        worst_line_number = list(sorted_line_number)
        sorted_list = hit_count[sorted_index]
        worse_list, current_score, start_index, oracle_lines = [], -1, -1, oracle_line_dict[target_file_name]
        for ii in range(len(sorted_list)):
            if sorted_list[ii] != current_score:
                current_score = sorted_list[ii]
                start_index = ii
            elif worst_line_number[ii] not in oracle_lines:
                temp = worst_line_number[ii]  # 取出这个无bug的行号
                for t in range(ii, start_index, -1):
                    worst_line_number[t] = worst_line_number[t - 1]
                worst_line_number[start_index] = temp
        worst_list_dict[target_file_name] = worst_line_number
        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################

        # ###################################### 切分点设置 ################################################
        # 20% effort (i.e, LOC)
        effort_cf_dict[target_file_name] = int(.2 * len(target_file_lines))
        # 设置分类切分点: 所有包含risk tokens (i.e., hit_count[i] > 0) 的代码行被预测为有 bug
        defect_cf_dict[target_file_name] = int(.5 * len(target_file_lines))
        # print('%d/%d files predicted finish!' % (i, len(defect_prone_file_indices)))

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)


# OK 进行代码行级别的排序
def LM_1_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return Ngram_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, 1)


# OK 进行代码行级别的排序
def LM_2_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return Ngram_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, 2)


# OK 进行代码行级别的排序
def LM_3_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return Ngram_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, 3)


# OK 进行代码行级别的排序
def LM_4_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    return Ngram_Model(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, 4)


def run_global_lm():
    n_gram_order = 2
    for project, releases in get_project_releases_dict().items():
        print(releases)
        print('Processing project ', project)
        for i in range(1, len(releases)):
            target_release = releases[i]
            build_global_language_model(target_release, n_gram_order)


# ========================================== cache ===========================================
def get_cache_tokenize(release, train_text, train_file_name, n):
    token_dict = {}
    for i in range(1, n + 1):

        for file_index in range(len(train_file_name)):
            target_file_name = train_file_name[file_index]
            target_file_text = train_text[file_index]

            ngram_vector = CountVectorizer(lowercase=False, stop_words=None, ngram_range=(i, i))
            corpus = target_file_text
            train_vtr = ngram_vector.fit_transform(corpus)
            text = '\n'.join([f'{token}:{train_vtr[0, index]}' for token, index in ngram_vector.vocabulary_.items()])

            # save_result(file_path, text)
            # print(f'{file_path} word output finish!')


def predict_cache_entropy(test_file, analysis, ngram_dict, prefix_dict, suffix_dict, order):
    entropy_of_each_file = []
    # process each file
    for line_index in range(len(test_file)):
        line = test_file[line_index]
        # process each line
        words_in_line = analysis(line.strip())
        # the number of words in the line
        num_of_words = len(words_in_line)
        # entropy = -1/n * sum( p(ti,h) * log(p(ti,h)) )
        entropy = -1
        if num_of_words > 0:
            ent_of_token = 0
            for i in range(num_of_words):
                if order == 1:
                    # n-gram n==1
                    prob = ngram_dict.get(words_in_line[i], ngram_dict['-'])
                else:
                    # n-gram n>=2
                    # prolog
                    start = 0 if i - (order - 1) < 0 else i - (order - 1)
                    # 前缀 当前词 当前词的概率
                    prefix, current, prob_prefix = ' '.join(words_in_line[start:i]), words_in_line[i], 0
                    if len(prefix) == 0:
                        prob_prefix = ngram_dict.get(current, ngram_dict['-'])
                    else:
                        if prefix not in prefix_dict:
                            prob_prefix = prefix_dict['-']
                        else:
                            prob_prefix = prefix_dict[prefix].get(current, prefix_dict[prefix]['-'])

                    # epilog
                    end = num_of_words if i + (order - 1) > num_of_words else i + (order - 1)
                    suffix, current, prob_suffix = ' '.join(words_in_line[i:end]), words_in_line[i], 0
                    if len(suffix) == 0:
                        prob_suffix = ngram_dict.get(current, ngram_dict['-'])
                    else:
                        if suffix not in suffix_dict:
                            prob_suffix = suffix_dict['-']
                        else:
                            prob_suffix = suffix_dict[suffix].get(current, suffix_dict[suffix]['-'])

                    prob = (prob_prefix + prob_suffix) / 2

                # ent_of_token += prob * math.log(prob)
                ent_of_token += math.log(prob)

            entropy = -(ent_of_token / num_of_words)
            # print(entropy)
        entropy_of_each_file.append(entropy)
    return np.array(entropy_of_each_file)


def build_cache_n_gram(target_release, n):
    for i in range(1, n + 1):
        if i == 1:
            # 序列字典
            seq_n_dict = count_sequence(target_release, i)
            # 字典大小
            size_of_v = len(seq_n_dict)
            # 所有序列出现的次数总和
            size_of_all_seq = sum([count for token, count in seq_n_dict.items()])
            # 未出现的词的概率
            n_gram_dict = {'-': 1 / (size_of_all_seq + size_of_v)}
            # 计算每个序列对应的概率
            for token in seq_n_dict.keys():
                n_gram_dict[token] = (seq_n_dict[token] + 1) / (size_of_all_seq + size_of_v)
            # 存储概率字典
            dump_pk_result(f'{result_path}NBF/{target_release}/n-gram.pk', n_gram_dict)
        else:
            n_gram_prefix_dict = {}
            n_gram_suffix_dict = {}
            # sequence with length of n, sequence with length of n-1
            seq_n_dict, seq_n_1_dict = count_sequence(target_release, i), count_sequence(target_release, i - 1)

            for seq, count_seq in seq_n_dict.items():
                # seq: 'java util HashMap' count_seq:37
                split = seq.split(' ')
                # 前缀序列和后缀序列
                seq_prefix, seq_suffix = ' '.join(split[:-1]), ' '.join(split[1:])
                # 前缀对应的尾部单词 后缀对应的首部单词
                token_prefix, token_suffix = split[-1], split[0]
                # 前缀序列的频率 后缀序列的频率 NOTE 前缀或者后缀一定存在
                count_prefix, count_suffix = seq_n_1_dict[seq_prefix], seq_n_1_dict[seq_suffix]
                # 词库的大小
                size_of_v = len(get_vocabulary(target_release))
                # 前缀常量 后缀常量
                prefix_c, suffix_c = 1 / (count_prefix + size_of_v), 1 / (count_suffix + size_of_v)
                # 获取 序列前缀的字典 序列后缀的字典
                d_prefix = n_gram_prefix_dict[seq_prefix] if seq_prefix in n_gram_prefix_dict else {'-': prefix_c}
                d_suffix = n_gram_suffix_dict[seq_suffix] if seq_suffix in n_gram_suffix_dict else {'-': suffix_c}

                d_prefix[token_prefix] = (count_seq + 1) / (count_prefix + size_of_v)
                d_suffix[token_suffix] = (count_seq + 1) / (count_suffix + size_of_v)

                n_gram_prefix_dict[seq_prefix] = d_prefix
                n_gram_suffix_dict[seq_suffix] = d_suffix

            dump_pk_result(f'{result_path}NBF/{target_release}/{i}-gram_prefix.pk', n_gram_prefix_dict)
            dump_pk_result(f'{result_path}NBF/{target_release}/{i}-gram_suffix.pk', n_gram_suffix_dict)
    print(f'==================== {n}-gram dict of {target_release} output finish ========================'[:80])


def build_cache_language_model(target_release, order):
    print(f'Building cache {order}-gram model for {target_release}')
    # read line level text from the target release
    train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(target_release)
    # tokenize a source file
    get_tokenize(target_release, train_text, order)
    build_cache_n_gram(target_release, order)
    print(f'The {order}-gram model has been built finish!')


def run_cache_lm():
    n_gram_order = 2
    for project, releases in get_project_releases_dict().items():
        print(releases)
        print('Processing project ', project)
        for i in range(1, len(releases)):
            target_release = releases[i]
            build_cache_language_model(target_release, n_gram_order)


if __name__ == '__main__':
    run_global_lm()
