# -*- coding:utf-8 -*-

import src.models.natural as natural
from src.exps.cross_release import *
from src.exps.within_release import *
from src.models.access import AccessModel
from src.models.line_dp import LineDPModel

# 忽略警告信息
from src.models.static_analysis_tools import detect_bugs_by_checkstyle_from_each_single_file

warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

# Global control variable
Predict = 'Predict'
Eval = 'Evaluate'


# ################# Run within release prediction experiments ###################
def run_within_release_prediction(prediction_model, mode=Predict):
    threshold = 50
    wp_result_path = '%sResult/WP/%s_%d/' % (root_path, getattr(prediction_model, '__name__'), threshold)
    make_path(wp_result_path)
    for project_release in get_project_release_list():
        if mode == Predict:
            predict_within_release(project_release, 10, 10, prediction_model, threshold, wp_result_path)
        else:
            eval_within_release(project_release, 10, 10, wp_result_path, depend=True)
    if mode == Eval:
        combine_within_results(wp_result_path)


# ################# Run cross release prediction experiments ###################
def run_cross_release_prediction(prediction_model, mode=Predict):
    # thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    thresholds = [50]
    for threshold in thresholds:
        cp_result_path = '%sResult/CP/%s_%d/' % (root_path, getattr(prediction_model, '__name__'), threshold)
        make_path(cp_result_path)
        for project, releases in get_project_releases_dict().items():
            if mode == Predict:
                predict_cross_release(project, releases, prediction_model, threshold, cp_result_path)
            else:
                eval_cross_release(project, releases, cp_result_path, depend=True)
        if mode == Eval:
            combine_cross_results(cp_result_path)


if __name__ == '__main__':
    # 运行版本内预测实验
    # run_within_release_prediction(AccessModel, Eval)
    # run_within_release_prediction(AccessModel, Eval)
    # 运行版本间预测实验
    # run_cross_release_prediction(AccessModel, Eval)

    natural.main()
    pass
