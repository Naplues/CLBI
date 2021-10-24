# -*- coding:utf-8 -*-
import math
import os

import pandas as pd

import numpy as np
from sklearn import metrics

from src.utils.helper import read_file_level_dataset, read_line_level_dataset, make_path, root_path


class BaseModel(object):
    threshold = 50
    model_name = 'BaseModel'  # need to be rewrite

    def __init__(self, train_release, test_release):
        # random seed is set as 0-9
        self.random_seed = 0
        np.random.seed(self.random_seed)

        self.iter_num = 100
        # Model configuration
        self.project_name = train_release.split('-')[0]

        # 训练和测试版本
        self.train_release = train_release
        self.test_release = test_release
        # file level data 文件级数据
        self.train_text, self.train_text_lines, self.train_label, self.train_filename = \
            read_file_level_dataset(train_release)
        self.test_text, self.test_text_lines, self.test_labels, self.test_filename = \
            read_file_level_dataset(test_release)

        # 明确存储实验结果的每个文件夹及文件路径
        # Specific the actual name of each folder or file
        self.cp_result_path = f'{root_path}Result/{self.model_name}/'
        self.buggy_density_path = f'{root_path}Result/buggy_density/'
        self.buggy_density_file = f'{self.buggy_density_path}density-{self.test_release}.csv'
        self.file_level_result_path = f'{self.cp_result_path}file_result/'
        self.line_level_result_path = f'{self.cp_result_path}line_result/'
        self.file_level_result_file = f'{self.file_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.line_level_result_file = f'{self.line_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.file_level_evaluation_file = f'{self.file_level_result_path}evaluation.csv'
        self.line_level_evaluation_file = f'{self.line_level_result_path}evaluation.csv'
        # 创建文件存储目录
        self.init_file_path()

        # File level data 文件级别数据 # 单独计算每种方法预测得到得缺陷密度
        self.test_pred_labels = []
        self.test_pred_scores = []
        self.test_pred_density = dict()

        # Line level data 代码行级数据
        self.oracle_line_dict = read_line_level_dataset(self.test_release)
        self.actual_buggy_lines = self.get_actual_buggy_lines()  # set
        self.predicted_buggy_lines = []
        self.predicted_buggy_score = []
        self.predicted_density = []

        self.num_actual_buggy_lines = 0
        self.num_predicted_buggy_lines = 0
        self.total_lines = 0
        self.total_lines_in_defective_files = 0

        self.rank_strategy = self.rank_strategy_1()
        print(f"Training set\t ===> {self.test_release}\tTest set.")

    def init_file_path(self):
        # 创建文件夹目录
        # Create directory for each folder
        make_path(self.cp_result_path)
        make_path(self.buggy_density_path)
        make_path(self.file_level_result_path)
        make_path(self.line_level_result_path)
        make_path(f'{self.file_level_result_path}{self.project_name}/')
        make_path(f'{self.line_level_result_path}{self.project_name}/')

    def get_buggy_density(self):
        buggy_density = []
        return buggy_density

    def get_actual_buggy_lines(self):
        oracle_line_list = set()
        for file_name in self.oracle_line_dict:
            oracle_line_list.update([f'{file_name}:{line}' for line in self.oracle_line_dict[file_name]])
        return oracle_line_list

    def file_level_prediction(self):
        pass

    def line_level_prediction(self):
        pass

    def analyze_file_level_result(self):
        """
        分析评估文件级的分类结果
        :return:
        """
        assert len(self.test_labels) == len(self.test_pred_labels), 'The lengths are not equal'

        total_file, identified_file, total_line, identified_line, predicted_file, predicted_line = 0, 0, 0, 0, 0, 0

        for index in range(len(self.test_labels)):
            buggy_line = len(self.test_text_lines[index])
            if self.test_pred_labels[index] == 1:
                predicted_file += 1
                predicted_line += buggy_line

        for index in range(len(self.test_labels)):
            if self.test_labels[index] == 1:
                buggy_line = len(self.oracle_line_dict[self.test_filename[index]])
                if self.test_pred_labels[index] == 1:
                    identified_line += buggy_line
                    identified_file += 1
                total_line += buggy_line
                total_file += 1

        print(f'Buggy file hit info: {identified_file}/{total_file} - {round(identified_file / total_file * 100, 1)}%')
        print(f'Buggy line hit info: {identified_line}/{total_line} - {round(identified_line / total_line * 100, 1)}%')
        print(f'Predicted {predicted_file} buggy files contain {predicted_line} lines')
        self.num_actual_buggy_lines = identified_line
        self.total_lines = sum([len(lines) for lines in self.test_text_lines])
        self.total_lines_in_defective_files = predicted_line

        # File level result
        if os.path.exists(self.file_level_result_file):
            return
        data = {'filename': self.test_filename,
                'oracle': self.test_labels,
                'predicted_label': self.test_pred_labels,
                'predicted_score': self.test_pred_scores}
        data = pd.DataFrame(data, columns=['filename', 'oracle', 'predicted_label', 'predicted_score'])
        data.to_csv(self.file_level_result_file, index=False)

        # File level evaluation
        append_title = True if not os.path.exists(self.file_level_evaluation_file) else False
        title = 'release,precision,recall,f1-score,accuracy,mcc,identified/total files,max identified/total lines\n'
        with open(self.file_level_evaluation_file, 'a') as file:
            file.write(title) if append_title else None
            file.write(f'{self.test_release},'
                       f'{metrics.precision_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.recall_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.f1_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.accuracy_score(self.test_labels, self.test_pred_labels)},'
                       f'{metrics.matthews_corrcoef(self.test_labels, self.test_pred_labels)},'
                       f'{identified_file}/{total_file},'
                       f'{identified_line}/{total_line},'
                       f'\n')
        return

    def analyze_line_level_result(self):
        def load_result_data():
            predicted_lines, predicted_score, predicted_density = [], [], []
            with open(self.line_level_result_file, 'r') as f:
                for l in f.readlines()[1:]:
                    predicted_lines.append(l.strip().split(',')[0])
                    predicted_score.append(float(l.strip().split(',')[1]))
                    predicted_density.append(float(l.strip().split(',')[2]))
            return predicted_lines, predicted_score, predicted_density

        self.predicted_buggy_lines, self.predicted_buggy_score, self.predicted_density = load_result_data()

        ######################### classification performance indicators #########################
        tp = len(self.actual_buggy_lines.intersection(self.predicted_buggy_lines))
        fp = len(self.predicted_buggy_lines) - tp
        # fn = len(self.actual_buggy_lines) - tp
        fn = self.num_actual_buggy_lines - tp
        # tn = self.total_lines - tp - fp - fn # 22 11366 191 183785
        tn = self.total_lines_in_defective_files - tp - fp - fn  # 22 11366 6 17000

        precision = .0 if tp + fp == .0 else tp / (tp + fp)
        recall = .0 if tp + fn == .0 else tp / (tp + fn)
        far = .0 if fp + tn == 0 else fp / (fp + tn)
        ce = .0 if fn + tn == .0 else fn / (fn + tn)

        d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(0 - far, 2)) / math.sqrt(2)
        mcc = .0 if tp + fp == .0 or tp + fn == .0 or tn + fp == .0 or tn + fn == .0 else \
            (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        ######################### ranking performance indicators #########################
        ifa, recall_20 = self.rank_strategy  # Strategy 1

        append_title = True if not os.path.exists(self.line_level_evaluation_file) else False
        title = 'release,precision,recall,far,ce,d2h,mcc\n'
        with open(self.line_level_result_file, 'a') as file:
            file.write(title) if append_title else None
            file.write(f'{self.test_release},{precision},{recall},{far},{ce},{d2h},{mcc},{ifa},{recall_20}\n')
        return

    def rank_strategy_1(self):
        ifa_list, recall_20_list = [], []
        for it in range(self.iter_num):
            np.random.seed(it)
            predicted_buggy_lines = self.predicted_buggy_lines
            predicted_buggy_score = [score + np.random.random() for score in self.predicted_buggy_score]
            sorted_index = np.argsort(predicted_buggy_score)[::-1]
            ranked_predicted_buggy_lines = np.array(predicted_buggy_lines)[sorted_index]

            count, ifa, recall_20, max_len = 0, 0, 0, int(self.total_lines_in_defective_files * 0.2)
            for line in ranked_predicted_buggy_lines[:max_len]:
                if line in self.actual_buggy_lines:
                    ifa = count if ifa == 0 else ifa
                    recall_20 += 1
                count += 1
            ifa_list.append(ifa)
            recall_20_list.append(recall_20)

        return np.mean(ifa_list), np.mean(recall_20_list) / self.num_actual_buggy_lines

    def rank_strategy_2(self):
        ranked_predicted_buggy_lines = []
        # 按照文件包含缺陷的概率从大到小排序
        # Indices of defective files in descending order according to the prediction scores
        defective_file_index = [i for i in np.argsort(self.test_pred_scores)[::-1] if self.test_pred_labels[i] == 1]
        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            temp_lines, temp_scores = [], []
            for index in range(len(self.predicted_buggy_lines)):
                if self.predicted_buggy_lines[index].startswith(defective_filename):
                    temp_lines.append(self.predicted_buggy_lines[index])
                    temp_scores.append(self.predicted_buggy_score[index])

            sorted_index = np.argsort(temp_scores)[::-1]
            ranked_temp_buggy_lines = list(np.array(temp_lines)[sorted_index])
            ranked_predicted_buggy_lines.extend(ranked_temp_buggy_lines)

        count, ifa, recall_20, max_len = 0, 0, 0, int(self.total_lines_in_defective_files * 0.2)
        for line in ranked_predicted_buggy_lines[:max_len]:
            if line in self.actual_buggy_lines:
                ifa = count if ifa == 0 else ifa
                recall_20 += 1
            count += 1
        return ifa, recall_20 / self.num_actual_buggy_lines

    def rank_strategy_3(self):
        ranked_predicted_buggy_lines = []
        # 按照文件包含缺陷的密度从大到小排序
        density_dict = dict()
        for index in range(len(self.predicted_buggy_lines)):
            filename = self.predicted_buggy_lines[index].split(':')[0]
            if filename not in density_dict:
                density_dict[filename] = self.predicted_density[index]
        test_pred_density = []
        for filename in self.test_filename:
            test_pred_density.append(density_dict[filename]) if filename in density_dict else 0

        # Indices of defective files in descending order according to the prediction scores
        defective_file_index = [i for i in np.argsort(test_pred_density)[::-1] if self.test_pred_labels[i] == 1]
        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            temp_lines, temp_scores = [], []
            for index in range(len(self.predicted_buggy_lines)):
                if self.predicted_buggy_lines[index].startswith(defective_filename):
                    temp_lines.append(self.predicted_buggy_lines[index])
                    temp_scores.append(self.predicted_buggy_score[index])

            sorted_index = np.argsort(temp_scores)[::-1]
            ranked_temp_buggy_lines = list(np.array(temp_lines)[sorted_index])
            ranked_predicted_buggy_lines.extend(ranked_temp_buggy_lines)

        count, ifa, recall_20, max_len = 0, 0, 0, int(self.total_lines_in_defective_files * 0.2)
        for line in ranked_predicted_buggy_lines[:max_len]:
            if line in self.actual_buggy_lines:
                ifa = count if ifa == 0 else ifa
                recall_20 += 1
            count += 1
        return ifa, recall_20 / self.num_actual_buggy_lines

    def rank_strategy_4(self):
        return .0, .0
