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


def count_sequence(train_release, n):
    seq_dict = {}
    lines = read_data_from_file(f'{result_path}NBF/{train_release}/{n}-gram.txt')
    for line in lines:
        split = line.split(':')
        seq, count = split[0], int(split[1])
        seq_dict[seq] = count
    return seq_dict


def train_n_gram(train_release, n):
    # 字典
    n_gram_prefix_dict = {}
    n_gram_suffix_dict = {}
    for i in range(1, n + 1):
        if i == 1:
            seq_n_dict = count_sequence(train_release, i)
            size_of_v = len(seq_n_dict)
            size = sum([count for token, count in seq_n_dict.items()])
            n_gram_dict = {'-': 1 / (size + size_of_v)}
            for token in seq_n_dict.keys():
                n_gram_dict[token] = (seq_n_dict[token] + 1) / (size + size_of_v)
            dump_pk_result(f'{result_path}NBF/{train_release}/n-gram.pk', n_gram_dict)
        else:
            seq_n_dict = count_sequence(train_release, i)
            seq_n_1_dict = count_sequence(train_release, i - 1)

            for seq, count_seq in seq_n_dict.items():
                split = seq.split(' ')
                seq_prefix, seq_suffix = ' '.join(split[:-1]), ' '.join(split[1:])
                token_prefix, token_suffix = split[-1], split[0]
                count_prefix, count_suffix = seq_n_1_dict[seq_prefix], seq_n_1_dict[seq_suffix]

                size_of_v = len(get_vocabulary(train_release))
                prefix_c, suffix_c = 1 / (count_prefix + size_of_v), 1 / (count_suffix + size_of_v)

                d_prefix = n_gram_prefix_dict[seq_prefix] if seq_prefix in n_gram_prefix_dict else {'-': prefix_c}
                d_suffix = n_gram_suffix_dict[seq_suffix] if seq_suffix in n_gram_suffix_dict else {'-': suffix_c}

                d_prefix[token_prefix] = (count_seq + 1) / (count_prefix + size_of_v)
                d_suffix[token_suffix] = (count_seq + 1) / (count_suffix + size_of_v)

                n_gram_prefix_dict[seq_prefix] = d_prefix
                n_gram_suffix_dict[seq_suffix] = d_suffix
            print(f'{i}-gram dict output finish')

            dump_pk_result(f'{result_path}NBF/{train_release}/{i}-gram_prefix.pk', n_gram_prefix_dict)
            dump_pk_result(f'{result_path}NBF/{train_release}/{i}-gram_suffix.pk', n_gram_suffix_dict)
    print(f'============================ n-gram dict output finish =============================')


def train_language_model(train_release, order):
    train_text, train_text_lines, train_label, train_filename = read_file_level_dataset(train_release)
    get_tokenize(train_release, train_text, order)
    train_n_gram(train_release, order)


def predict_prob(train_release, test_release, order):
    result_dict = {}

    analysis = CountVectorizer(lowercase=False, stop_words=None).build_analyzer()
    ngram_dict = load_pk_result(f'{result_path}NBF/{train_release}/n-gram.pk')
    prefix_dict = load_pk_result(f'{result_path}NBF/{train_release}/n-gram_prefix.pk')
    suffix_dict = load_pk_result(f'{result_path}NBF/{train_release}/n-gram_suffix.pk')

    test_text, test_text_lines, test_label, test_filename = read_file_level_dataset(test_release)
    for index in range(len(test_text_lines)):
        result_of_test_file = []
        # process each file
        for line_index in range(len(test_text_lines[index])):
            line = test_text_lines[index][line_index]
            # process each line
            words_in_line = analysis(line.strip())
            n = len(words_in_line)
            # entropy = -1/n * sum(log(p(ti,h)))
            if n == 0:
                entropy = -1
            else:
                ent_of_token = 0
                for i in range(1, n):
                    # prolog
                    start = 0 if i - (order - 1) < 0 else i - (order - 1)
                    prefix, current, prob_prefix = ' '.join(words_in_line[start:i]), words_in_line[i], 0
                    if len(prefix) == 0:
                        prob_prefix = ngram_dict.get(current, ngram_dict['-'])
                    else:
                        if prefix not in prefix_dict:
                            prob_prefix = prefix_dict['-']
                        else:
                            prob_prefix = prefix_dict[prefix].get(current, prefix_dict[prefix]['-'])

                    # epilog
                    end = n if i + (order - 1) > n else i + (order - 1)
                    suffix, current, prob_suffix = ' '.join(words_in_line[i:end]), words_in_line[i], 0
                    if len(suffix) == 0:
                        prob_suffix = ngram_dict.get(current, ngram_dict['-'])
                    else:
                        if suffix not in suffix_dict:
                            prob_suffix = suffix_dict['-']
                        else:
                            prob_suffix = suffix_dict[suffix].get(current, suffix_dict[suffix]['-'])

                    ent_of_token += math.log((prob_prefix + prob_suffix) / 2)
                entropy = -ent_of_token / n

            result_of_test_file.append(entropy)
        result_dict[test_filename[index]] = result_of_test_file
        break


def main():
    order = 6
    for project, releases in get_project_releases_dict().items():
        print('Processing project ', project)
        for i in range(len(releases) - 1):
            train_release, test_release = releases[i], releases[i + 1]
            print(f'{train_release}   ===>   {test_release}')
            # 训练语言模型
            train_language_model(train_release, order)
            # 测试语言模型
            # predict_prob(train_release, test_release, order)


if __name__ == '__main__':
    main()
