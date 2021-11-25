# -*- coding:utf-8 -*-
from src.models.explain import *
from src.models.glance import *
from src.models.natural import *
from src.models.tools import *
from src.utils.helper import *
from numpy import *


def select_models(exp: str = 'RQ2'):
    """
    Select proper models according to different purposes.
    :param exp: Experiment name
    :return: A list of model instances.
    """
    if exp == 'RQ2':
        m = [NGram(), LineDP(), ]
    elif exp == 'RQ3':
        m = [PMD(), CheckStyle()]
    else:
        m = []
    return m


def get_tp_buggy_lines(model, project, release):
    model.project_name, model.test_release = project, release

    model.line_level_result_file = f'{model.line_level_result_path}{model.project_name}/{model.test_release}-result.csv'
    model.predicted_buggy_lines = []
    model.load_line_level_result()
    predicted_buggy_lines = model.predicted_buggy_lines

    model.oracle_line_dict = dict()
    actual_buggy_lines = model.get_actual_buggy_lines()

    tp_buggy_lines = actual_buggy_lines.intersection(predicted_buggy_lines)

    return tp_buggy_lines


def classification_difference(exp="RQ2"):
    glance_models = [Glance_MD(), Glance_EA(), Glance_LR()]
    target_models = select_models(exp)
    text = ''
    for glance_model in glance_models:
        for target_model in target_models:

            tp_data, hit_data, over_data = '', '', ''
            hit_list, over_list = [], []

            for project, releases in get_project_releases_dict().items():
                print(glance_model.model_name, target_model.model_name, project)
                release_tp_data, release_hit_data, release_over_data = project + ',', project + ',', project + ','
                for test_release in releases[1:]:
                    target_buggy_lines = get_tp_buggy_lines(target_model, project, test_release)
                    glance_buggy_lines = get_tp_buggy_lines(glance_model, project, test_release)

                    tp_target = len(target_buggy_lines)
                    num_hit = len(glance_buggy_lines.intersection(target_buggy_lines)) / tp_target
                    num_over = len(glance_buggy_lines - target_buggy_lines) / tp_target

                    release_tp_data += str(tp_target) + ','
                    release_hit_data += str(round(num_hit, 3)) + ','
                    release_over_data += str(round(num_over, 3)) + ','

                    hit_list.append(num_hit)
                    over_list.append(num_over)

                tp_data += release_tp_data + '\n'
                hit_data += release_hit_data + '\n'
                over_data += release_over_data + '\n'

            save_csv_result(f'../../result/{exp}/Difference/',
                            f'{glance_model.model_name}-{target_model.model_name}_TP_data.csv', tp_data)
            save_csv_result(f'../../result/{exp}/Difference/',
                            f'{glance_model.model_name}-{target_model.model_name}_Hit_data.csv', hit_data)
            save_csv_result(f'../../result/{exp}/Difference/',
                            f'{glance_model.model_name}-{target_model.model_name}_Over_data.csv', over_data)

            text += f'{glance_model.model_name},{target_model.model_name},{mean(hit_list)},{mean(over_list)}\n'
    save_csv_result(f'../../result/{exp}/', 'Difference_summary.csv', text)


def diff_ranking():
    access_model = 'AccessModel'
    target_model = 'CheckStyleModel'  # PMDModel CheckStyleModel
    text = ''
    for project, releases in get_project_releases_dict().items():

        print(project, end=',')
        increase, decrease = [], []
        for release in releases[1:]:
            data_access = load_pk_result(f'{result_path}Diff_Ranking/{access_model}/{release}.pk')
            data_target = load_pk_result(f'{result_path}Diff_Ranking/{target_model}/{release}.pk')
            i, d = [], []
            t, text_lines, label, filename = read_file_level_dataset(release)
            len_dict = {}
            for index in range(len(filename)):
                len_dict[filename[index]] = len(text_lines[index])

            for file_line in data_target.keys():
                if file_line not in data_access.keys():
                    continue
                rank_of_access = data_access[file_line]
                rank_of_target = data_target[file_line]
                # increase
                if rank_of_access < rank_of_target:
                    i.append((rank_of_target - rank_of_access) / len_dict[file_line.split(':')[0]])
                # decrease
                if rank_of_access > rank_of_target:
                    d.append((rank_of_access - rank_of_target) / len_dict[file_line.split(':')[0]])

            increase.append(mean(i))
            decrease.append(mean(d))

        text += project + ',' + str(increase).replace('[', '').replace(']', '') + '\n'
        text += project + ',' + str(decrease).replace('[', '').replace(']', '') + '\n'
    print(text)
    save_result('C:/Users/GZQ/Desktop/d.csv', text)


if __name__ == '__main__':
    # diff_ranking()
    for experiment in ["RQ2", "RQ3"]:
        classification_difference(experiment)

        pass
    pass
