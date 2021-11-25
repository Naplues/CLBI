# -*- coding:utf-8 -*-

import math

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.config import USE_CACHE
from src.utils.helper import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class BaseModel(object):
    model_name = 'BaseModel'  # need to be rewrite in subclass

    def __init__(self, train_release: str = '', test_release: str = '', test_result_path=''):
        # Specific the actual name of each folder or file

        self.result_path = f'{root_path}Result/{self.model_name}/'
        if test_result_path != '':
            self.result_path = test_result_path
        # folder path
        self.file_level_result_path = f'{self.result_path}file_result/'
        self.line_level_result_path = f'{self.result_path}line_result/'
        self.buggy_density_path = f'{self.result_path}buggy_density/'
        # evaluation path
        self.file_level_evaluation_file = f'{self.file_level_result_path}evaluation.csv'
        self.line_level_evaluation_file = f'{self.line_level_result_path}evaluation.csv'

        # Model configuration info
        self.project_name = train_release.split('-')[0]
        np.random.seed(0)
        self.random_state = 0
        self.iter_num = 100
        self.threshold_effort = 0.2

        # Model training and test data
        self.train_release = train_release
        self.test_release = test_release
        # File level classifier
        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

        # file level data 文件级数据
        self.train_text, self.train_text_lines, self.train_label, self.train_filename = \
            read_file_level_dataset(train_release)
        self.test_text, self.test_text_lines, self.test_labels, self.test_filename = \
            read_file_level_dataset(test_release)

        # 明确存储实验结果的每个文件夹及文件路径
        # result file path
        self.file_level_result_file = f'{self.file_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.line_level_result_file = f'{self.line_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.buggy_density_file = f'{self.buggy_density_path}{self.test_release}-density.csv'
        self.commit_buggy_path = f'{root_path}/Dataset/Bug-Info/{self.test_release.split("-")[0]}'

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

        self.num_total_lines = sum([len(lines) for lines in self.test_text_lines])
        self.num_actual_buggy_lines = len(self.actual_buggy_lines)

        # NOTE need to be written in the method of line_level_prediction of subclass
        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

        self.rank_strategy = 3  # 1, 2, 3,

    def init_file_path(self):
        # 创建文件夹目录
        # Create directory for each folder
        make_path(self.result_path)
        make_path(self.file_level_result_path)
        make_path(self.line_level_result_path)
        make_path(self.buggy_density_path)
        make_path(f'{self.file_level_result_path}{self.project_name}/')
        make_path(f'{self.line_level_result_path}{self.project_name}/')

    def get_actual_buggy_lines(self):
        if len(self.oracle_line_dict.keys()) == 0:
            self.oracle_line_dict = read_line_level_dataset(self.test_release)
        oracle_line_list = set()
        for file_name in self.oracle_line_dict:
            oracle_line_list.update([f'{file_name}:{line}' for line in self.oracle_line_dict[file_name]])
        return oracle_line_list

    # ====================================================================================================
    # Buggy files and lines prediction
    # ====================================================================================================
    def file_level_prediction(self):
        """
        NOTE: This method should be implemented by sub class (i.e., Glance)
        """
        print(f"Prediction\t=>\t{self.test_release}")
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        # 2. Convert text feature into numerical feature, classifier
        # Neither perform lowercase, stemming, nor lemmatization. Remove tokens that appear only once
        train_vtr = self.vector.fit_transform(self.train_text)
        test_vtr = self.vector.transform(self.test_text)
        # 3. Predict defective files, test_predictions
        self.clf.fit(train_vtr, self.train_label)

        self.test_pred_labels = self.clf.predict(test_vtr)
        # Obtain the prediction scores of each buggy lines.
        if self.model_name == 'TMI-SVM':  # NOTE may be removed later
            self.test_pred_scores = np.array([score for score in self.test_pred_labels])
        else:
            self.test_pred_scores = np.array([score[1] for score in self.clf.predict_proba(test_vtr)])

        # Save file level result
        self.save_file_level_result()

    def line_level_prediction(self):
        """
        NOTE: This method must be implemented by sub class
        """
        print(f'Line level prediction for: {self.model_name}')
        pass

    # ====================================================================================================
    # Analyze the predicted buggy files and lines
    # NOTE: The below methods should be inherited rather than reimplemented by sub class.
    # ====================================================================================================
    def analyze_file_level_result(self):
        """
        Checked OK.
        Require: self.test_pred_labels
        Require: self.test_pred_scores
        """
        self.load_file_level_result()

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

        # File level evaluation append file
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
        """
        Checked OK.
        Require: self.test_pred_labels
        Require: self.predicted_buggy_lines
        Require: self.predicted_buggy_score
        Require: self.predicted_density
        Require: self.num_total_lines
        Require: self.num_actual_buggy_lines
        """
        ################################# Loading file level prediction result #################################
        total_lines_in_defective_files, buggy_lines_in_defective_files = 0, 0

        for index in range(len(self.test_pred_labels)):
            # Statistics in predicted defective files
            if self.test_pred_labels[index] == 1:
                total_lines_in_defective_files += len(self.test_text_lines[index])
                if self.test_labels[index] == 1:
                    buggy_lines_in_defective_files += len(self.oracle_line_dict[self.test_filename[index]])

        # TODO The below assignment statements should be commented
        # self.num_total_lines = total_lines_in_defective_files  # TP + FP + TN + FN
        # self.num_actual_buggy_lines = buggy_lines_in_defective_files  # TP + FN
        self.load_file_level_result()
        ################################# Loading predicted lines, scores, and density #################################
        self.load_line_level_result()

        ######################### Classification performance Performance Performance Indicators ################################################
        tp = len(self.actual_buggy_lines.intersection(self.predicted_buggy_lines))
        fp = self.num_predict_buggy_lines - tp
        fn = self.num_actual_buggy_lines - tp
        tn = self.num_total_lines - tp - fp - fn
        print(f'Total lines: {self.num_total_lines}\n'
              f'Buggy lines: {self.num_actual_buggy_lines}\n'
              f'Predicted lines: {len(self.predicted_buggy_lines)}\n'
              f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')

        prec = .0 if tp + fp == .0 else tp / (tp + fp)
        recall = .0 if tp + fn == .0 else tp / (tp + fn)
        far = .0 if fp + tn == 0 else fp / (fp + tn)
        ce = .0 if fn + tn == .0 else fn / (fn + tn)

        d2h = math.sqrt(math.pow(1 - recall, 2) + math.pow(0 - far, 2)) / math.sqrt(2)
        mcc = .0 if tp + fp == .0 or tp + fn == .0 or tn + fp == .0 or tn + fn == .0 else \
            (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        x, y = tp + fp, tp
        n, N = self.num_actual_buggy_lines, self.num_total_lines

        ER = .0 if (y * N) == .0 else (y * N - x * n) / (y * N)
        RI = .0 if (x * n) == 0 else (y * N - x * n) / (x * n)

        ################################ Ranking performance Performance Performance Indicators ################################################
        ifa, r_20 = .0, .0
        if self.rank_strategy == 1:
            ifa, r_20 = self.rank_strategy_1()  # Strategy 1
        elif self.rank_strategy == 2:
            ifa, r_20 = self.rank_strategy_2()  # Strategy 1
        elif self.rank_strategy == 3:
            ifa, r_20 = self.rank_strategy_3()  # Strategy 1

        ################################ Bug hit ratio ################################################
        buggy_lines_dict = read_dict_from_file(f'{self.commit_buggy_path}/{self.test_release}_commit_buggy_lines.csv')
        total_bugs = len(buggy_lines_dict.keys())
        hit_bugs = set()
        for line in self.predicted_buggy_lines:
            for bug_commit, lines in buggy_lines_dict.items():
                if line in lines:
                    hit_bugs.add(bug_commit)

        ratio = round(len(hit_bugs) / total_bugs, 3)

        ################################ Output the evaluation result ################################################
        append_title = True if not os.path.exists(self.line_level_evaluation_file) else False
        title = 'release,precision,recall,far,ce,d2h,mcc,ifa,recall_20,ER,RI,ratio\n'
        with open(self.line_level_evaluation_file, 'a') as file:
            file.write(title) if append_title else None
            file.write(f'{self.test_release},{prec},{recall},{far},{ce},{d2h},{mcc},{ifa},{r_20},{ER},{RI},{ratio}\n')
        return

    def rank_strategy_1(self):
        """
        Checked OK.
        Rank all predicted buggy lines according to the buggy scores.
        Require: self.iter_num
        Require: self.predicted_buggy_lines
        Require: self.predicted_buggy_score
        :return:
        """
        ifa_list, recall_20_list = [], []
        max_len = 0
        for it in range(self.iter_num):
            np.random.seed(it)
            predicted_buggy_score = [score + np.random.random() for score in self.predicted_buggy_score]
            sorted_index = np.argsort(predicted_buggy_score)[::-1]
            ranked_predicted_buggy_lines = np.array(self.predicted_buggy_lines)[sorted_index]

            max_len = len(ranked_predicted_buggy_lines)
            ifa, recall_20 = self.get_rank_performance(ranked_predicted_buggy_lines)
            ifa_list.append(ifa)
            recall_20_list.append(recall_20)

        max_effort = int(self.num_total_lines * self.threshold_effort)
        print(f'Predicted lines: {max_len}, Max effort: {max_effort}\n')
        return np.mean(ifa_list), np.mean(recall_20_list)

    def rank_strategy_2(self):
        """
        Checked OK.
        Two-stage ranking strategy.
        (1) Rank all predicted defective files according to the buggy score of each file;
        (2) Rank all buggy lines according to the score of each line in each file.
        Require: self.test_pred_scores
        Require: self.predicted_buggy_lines
        Require: self.predicted_buggy_score
        Require: self.num_total_lines
        :return: ifa, recall_20
        """
        ranked_predicted_buggy_lines = []
        # A list of buggy score for each file in test set.
        test_pred_score = self.test_pred_scores

        # Indices of defective files in descending order according to the prediction density.
        defective_file_index = [i for i in np.argsort(test_pred_score)[::-1] if self.test_pred_labels[i] == 1]

        for i in range(len(defective_file_index)):
            # The filename of predicted defective files in test set.
            defective_filename = self.test_filename[defective_file_index[i]]

            temp_lines, temp_scores = [], []
            for index in range(len(self.predicted_buggy_lines)):
                if self.predicted_buggy_lines[index].startswith(defective_filename):
                    temp_lines.append(self.predicted_buggy_lines[index])
                    temp_scores.append(self.predicted_buggy_score[index])

            # Indices of buggy lines in descending order according to scores in each file.
            sorted_index = np.argsort(temp_scores)[::-1]
            ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        # Calculate the performance values of IFA and Recall@20%.
        return self.get_rank_performance(ranked_predicted_buggy_lines)

    def rank_strategy_3(self):
        """
        Checked OK.
        Two-stage ranking strategy.
        (1) Rank all predicted defective files according to the buggy density;
        (2) Rank all buggy lines according to the score of each line in each file.
        Require: self.test_pred_density
        Require: self.predicted_buggy_lines
        Require: self.predicted_buggy_score
        Require: self.num_total_lines
        :return: ifa, recall_20
        """
        ranked_predicted_buggy_lines = []
        # A list of buggy density for each file in test set.
        test_pred_density = [self.test_pred_density[filename] for filename in self.test_filename]

        # Indices of defective files in descending order according to the prediction density.
        defective_file_index = [i for i in np.argsort(test_pred_density)[::-1] if self.test_pred_labels[i] == 1]

        for i in range(len(defective_file_index)):
            # The filename of predicted defective files in test set.
            defective_filename = self.test_filename[defective_file_index[i]]

            temp_lines, temp_scores = [], []
            for index in range(len(self.predicted_buggy_lines)):
                if self.predicted_buggy_lines[index].startswith(defective_filename):
                    temp_lines.append(self.predicted_buggy_lines[index])
                    temp_scores.append(self.predicted_buggy_score[index])

            # Indices of buggy lines in descending order according to scores in each file.
            sorted_index = np.argsort(temp_scores)[::-1]
            ranked_predicted_buggy_lines.extend(list(np.array(temp_lines)[sorted_index]))

        max_effort = int(self.num_total_lines * self.threshold_effort)
        print(f'Predicted lines: {len(ranked_predicted_buggy_lines)}, Max effort: {max_effort}\n')

        # Calculate the performance values of IFA and Recall@20%.
        return self.get_rank_performance(ranked_predicted_buggy_lines)

    def get_rank_performance(self, ranked_predicted_buggy_lines):
        """
        Require: self.num_total_lines
        Require: self.threshold_effort
        Require: self.actual_buggy_lines
        Require: self.num_actual_buggy_lines
        :param ranked_predicted_buggy_lines:
        :return:
        """
        count, ifa, recall_20, max_effort = 0, 0, 0, int(self.num_total_lines * self.threshold_effort)

        for line in ranked_predicted_buggy_lines[:max_effort]:
            if line in self.actual_buggy_lines:
                ifa = count if ifa == 0 else ifa
                recall_20 += 1
            count += 1
        return ifa, recall_20 / self.num_actual_buggy_lines

    # ============================================ File operation ======================================================
    def save_file_level_result(self):
        """
        Checked OK.
        Require: self.test_filename
        Require: self.test_labels
        Require: self.test_pred_labels
        Require: self.test_pred_scores
        Require: self.file_level_result_file
        :return:
        """
        data = {'filename': self.test_filename,
                'oracle': self.test_labels,
                'predicted_label': self.test_pred_labels,
                'predicted_score': self.test_pred_scores}
        data = pd.DataFrame(data, columns=['filename', 'oracle', 'predicted_label', 'predicted_score'])
        data.to_csv(self.file_level_result_file, index=False)

    def save_line_level_result(self):
        """
        Checked OK.
        Require: self.predicted_buggy_lines
        Require: self.predicted_buggy_score
        Require: self.predicted_density
        Require: self.line_level_result_file
        :return:
        """
        data = {'predicted_buggy_lines': self.predicted_buggy_lines,
                'predicted_buggy_score': self.predicted_buggy_score,
                'predicted_density': self.predicted_density}  # NOTE may be removed later
        data = pd.DataFrame(data, columns=['predicted_buggy_lines', 'predicted_buggy_score', 'predicted_density'])
        data.to_csv(self.line_level_result_file, index=False)

    def save_buggy_density_file(self):
        """
        Checked OK.
        Require: self.predicted_buggy_lines
        Require: self.predicted_buggy_score
        Require: self.predicted_density
        Require: self.line_level_result_file
        :return:
        """
        df = pd.read_csv(self.line_level_result_file)
        self.predicted_buggy_lines = list(df['predicted_buggy_lines'])  # only buggy lines in the defective files

        buggy_density, file_buggy_lines_dict = dict(), dict()
        # 读取预测为buggy的每个文件中包含的buggy代码行数
        for line in self.predicted_buggy_lines:
            filename = line.strip().split(':')[0]
            if not filename in file_buggy_lines_dict:
                file_buggy_lines_dict[filename] = 1
            else:
                file_buggy_lines_dict[filename] += 1
        # 计算缺陷密度 buggy lines/total lines
        for index in range(len(self.test_text_lines)):
            filename = self.test_filename[index]
            # 预测为无缺陷的文件 缺陷密度为0
            if filename not in file_buggy_lines_dict or len(self.test_text_lines[index]) == 0:
                buggy_density[filename] = 0
            else:
                buggy_density[filename] = file_buggy_lines_dict[filename] / len(self.test_text_lines[index])

        self.test_pred_density = buggy_density

        data = {'test_pred_density': self.test_pred_density}
        data = pd.DataFrame(data, columns=['test_pred_density'])
        data.to_csv(self.buggy_density_file, index=False)

    def load_file_level_result(self):
        # Load file prediction result if no result
        if len(self.test_pred_labels) == 0 or len(self.test_pred_scores) == 0:
            df = pd.read_csv(self.file_level_result_file)
            self.test_pred_labels = np.array(df['predicted_label'])
            self.test_pred_scores = np.array(df['predicted_score'])

    def load_line_level_result(self):
        if len(self.predicted_buggy_lines) == 0:
            df = pd.read_csv(self.line_level_result_file)
            self.predicted_buggy_lines = list(df['predicted_buggy_lines'])  # only buggy lines in the defective files
            self.predicted_buggy_score = list(df['predicted_buggy_score'])
            self.predicted_density = list(df['predicted_density'])
            self.num_predict_buggy_lines = len(self.predicted_buggy_lines)  # Require in the super class.
            self.save_buggy_density_file()
