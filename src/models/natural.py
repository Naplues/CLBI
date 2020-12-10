# -*- coding:utf-8 -*-

# n-gram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from src.utils.helper import *
import math


def get_tokenize(release, train_text, n):
    for i in range(1, n + 1):
        file_path = f'{result_path}NBF/{release}/{i}-gram.txt'
        if os.path.exists(file_path):
            continue
        ngram_vector = CountVectorizer(lowercase=False, stop_words=None, ngram_range=(i, i))
        corpus = [' '.join(train_text)]
        train_vtr = ngram_vector.fit_transform(corpus)
        text = '\n'.join([f'{token}:{train_vtr[0, index]}' for token, index in ngram_vector.vocabulary_.items()])

        save_csv_result(file_path, text)
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


def build_n_gram(target_release, n):
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


def build_language_model(target_release, order):
    print(f'Building {order}-gram model for {target_release}')
    # read line level text from the target release
    train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(target_release)
    # tokenize a source file
    get_tokenize(target_release, train_text, order)
    build_n_gram(target_release, order)
    print(f'The {order}-gram model has been built finish!')


def predict_prob(target_release, order):
    print(f'Predicting the entropy of {order}-gram model!')
    result_text = ''

    analysis = CountVectorizer(lowercase=False, stop_words=None).build_analyzer()
    ngram_dict = load_pk_result(f'{result_path}NBF/{target_release}/n-gram.pk')

    prefix_dict = load_pk_result(f'{result_path}NBF/{target_release}/{order}-gram_prefix.pk')
    suffix_dict = load_pk_result(f'{result_path}NBF/{target_release}/{order}-gram_suffix.pk')

    test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(target_release)
    for file_index in range(len(test_text_lines)):
        result_of_test_file = []
        # process each file
        for line_index in range(len(test_text_lines[file_index])):
            line = test_text_lines[file_index][line_index]
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

                    ent_of_token += prob * math.log(prob)
                entropy = -(ent_of_token / num_of_words)

            result_of_test_file.append(str(entropy))
        result_text += f'{test_filename[file_index]}:{",".join(result_of_test_file)}\n'
        save_csv_result(f'{result_path}LM-{order}/{target_release}.txt', result_text)


def run_lm():
    order = 1
    for project, releases in get_project_releases_dict().items():
        # if project != 'groovy':
        #    continue
        print(releases)
        print('Processing project ', project)
        for i in range(1, len(releases)):
            target_release = releases[i]
            # if target_release != 'groovy-1.5.5':
            #    continue
            build_language_model(target_release, order)
            # 测试语言模型
            # predict_prob(target_release, order)


if __name__ == '__main__':
    run_lm()
