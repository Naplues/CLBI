# -*- coding:utf-8 -*-
import math

import pandas as pd

from sklearn import metrics

from src.models.base_model import BaseModel
from src.utils.helper import *
from src.utils.eval import evaluation


def call_number(statement):
    statement = statement.strip('\"')
    score = 0
    for char in statement:
        if char == '(':
            score += 1
    return score


class Glance(BaseModel):
    model_name = 'Glance'

    def __init__(self, train_release, test_release):
        super().__init__(train_release, test_release)

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.tokenizer = self.vector.build_tokenizer()
        self.tags = ['todo', 'hack', 'fixme', 'xxx']
        # classifier

    def file_level_prediction(self):
        """
        Effort-Aware ManualDown File-level defect prediction
        :return:
        """
        print(f'Predicting ===> \t{self.test_release}')

        num_of_files = len(self.test_text)
        test_prediction = np.zeros(num_of_files, dtype=int).tolist()  # 初始化空的list

        # 每个文件的非空代码行, 自承认技术债, 分数
        loc, debts, score = [], [], []
        for file_index in range(num_of_files):
            # 所有测试文件的代码行组成的列表
            loc.append(len([line for line in self.test_text_lines[file_index] if line.strip() != '']))
            # 所有测试文件包含的SATD数组成的列表
            debts.append(len([tag for tag in self.tags if tag in self.test_text[file_index].lower()]))

        # 不包含技术债的文件排除掉
        score = loc
        # for file_index in range(len(text_lines)):
        #     if debts[file_index] == 0:
        #         score[file_index] = loc[file_index] / 2
        #
        # res = 'effort,debts,labels\n'
        # for index in range(len(text)):
        #     res += f'{loc[index]},{debts[index]},{labels[index]}\n'
        # save_csv_result(f'{root_path}temp/', f'{release}.csv', res)

        # 全部工作量 和 累积工作量
        effort_all, effort_acc = sum(loc), 0
        # 降序排列索引
        sorted_index = np.argsort(score).tolist()[::-1]

        file_count = 0
        for index in sorted_index:
            if effort_acc < effort_all * 0.5:
                # if count <= len(loc) / 2:
                test_prediction[index] = 1
                effort_acc += loc[index]
                file_count += 1
            else:
                break

        self.test_pred_labels = test_prediction
        self.test_pred_scores = np.array(score)

    def line_level_prediction(self):
        predicted_lines, predicted_score, predicted_density, total_lines = [], [], [], 0
        # Indices of defective files in descending order according to the prediction scores
        defective_file_index = [i for i in np.argsort(self.test_pred_scores)[::-1] if self.test_pred_labels[i] == 1]

        # 对预测为有bug的文件逐个进行代码行级别的排序
        for i in range(len(defective_file_index)):
            defective_filename = self.test_filename[defective_file_index[i]]
            # 有的测试文件(被预测为有bug,但实际上)没有bug,因此不会出现在 oracle 中,FP,这类文件要剔除,字典值为[]
            if defective_filename not in self.oracle_line_dict:
                self.oracle_line_dict[defective_filename] = []
            # 目标文件的代码行列表
            defective_file_line_list = self.test_text_lines[defective_file_index[i]]
            total_lines += len(defective_file_line_list)
            # ############################ 重点,怎么给每行赋一个缺陷值 ################################
            # 计算 每一行的权重, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数
            num_of_lines = len(defective_file_line_list)
            hit_count = np.zeros(num_of_lines, dtype=int)

            for line_index in range(num_of_lines):
                tokens_in_line = self.tokenizer(defective_file_line_list[line_index])
                if len(tokens_in_line) == 0:
                    hit_count[line_index] = 0
                else:
                    hit_count[line_index] = len(tokens_in_line) * call_number(defective_file_line_list[line_index]) + 1

                weight = 2

                if 'for' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'while' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'do' in tokens_in_line:
                    hit_count[line_index] *= weight

                if 'if' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'else' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'switch' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'case' in tokens_in_line:
                    hit_count[line_index] *= weight

                if 'continue' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'break' in tokens_in_line:
                    hit_count[line_index] *= weight
                if 'return' in tokens_in_line:
                    hit_count[line_index] *= weight

                # hit_count[line_index] = (hit_count[line_index] + 1)  # * lm_score[index]

            # ############################ 重点,怎么给每行赋一个缺陷值 ################################
            # 根据命中次数对代码行进行降序排序, 按照排序后数值从大到小的顺序显示代码行在原列表中的索引, cut_off 为切分点
            # line + 1,因为下标是从0开始计数而不是从1开始
            predicted_score.extend([hit_count[i] for i in range(num_of_lines) if hit_count[i] > 0])
            predicted_lines.extend(
                [f'{defective_filename}:{i + 1}' for i in range(num_of_lines) if hit_count[i] > 0])
            density = f'{len(np.where(hit_count > 0)) / len(hit_count)}'
            predicted_density.extend([density for i in range(num_of_lines) if hit_count[i] > 0])

        self.predicted_buggy_lines = predicted_lines
        self.predicted_buggy_score = predicted_score
        self.predicted_density = predicted_density
        self.total_lines_in_defective_files = total_lines

        # Line level result
        data = {'predicted_buggy_lines': self.predicted_buggy_lines,
                'predicted_buggy_score': self.predicted_buggy_score,
                'predicted_density': self.predicted_density}
        data = pd.DataFrame(data, columns=['predicted_buggy_lines', 'predicted_buggy_score', 'predicted_density'])
        data.to_csv(self.line_level_result_file, index=False)
