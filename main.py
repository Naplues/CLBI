# -*- coding:utf-8 -*-

import src.models.natural as natural
from src.exps.cross_release import *
from src.exps.within_release import *
from src.models.access import AccessModel
from src.models.explain import *

# 忽略警告信息
from src.models.static_analysis_tools import detect_bugs_by_checkstyle_from_each_single_file, PMDModel, CheckStyleModel

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
                # depend = True 时, Access 结果 依赖baseline方法, = False 时, 为自身的预测结果
                eval_cross_release(project, releases, cp_result_path, prediction_model, depend=False)
        if mode == Eval:
            combine_cross_results(cp_result_path)


if __name__ == '__main__':
    # 运行版本内预测实验
    # run_within_release_prediction(AccessModel, Eval)
    # run_within_release_prediction(AccessModel, Eval)
    # 运行版本间预测实验
    # run_cross_release_prediction(PMDModel, Eval)
    # run_cross_release_prediction(CheckStyleModel, Predict)

    # run_cross_release_prediction(TMI_LR_Model, Predict)
    # run_cross_release_prediction(TMI_SVM_L_Model, Predict)
    run_cross_release_prediction(TMI_MNB_Model, Predict)

    # run_cross_release_prediction(TMI_RF_Model, Predict)
    # run_cross_release_prediction(TMI_Tree_Model, Predict)
    pass

'''
camel-2.11.0 camel-2.12.0 camel-2.13.0 camel-2.14.0 camel-2.17.0 camel-2.18.0 camel-2.19.0
flink-1.4.0 flink-1.6.0
nifi-0.4.0 nifi-1.2.0 nifi-1.5.0 nifi-1.8.0 
ww-2.2.0 ww-2.2.2 ww-2.3.1 ww-2.3.4 ww-2.3.10 ww-2.3.15 ww-2.3.17 ww-2.3.20 ww-2 ww-2.3.24 
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.2.0')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.2.2')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.1')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.4')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.10')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.15')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.17')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.20')
    detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.3.24')
'''
