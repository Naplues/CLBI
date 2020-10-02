# -*- coding:utf-8 -*-

from exps.cross_release import *
from exps.within_release import *
from models.access import AccessModel
from models.line_dp import LineDPModel

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

root_path = r'C://Users/GZQ/Desktop/CLDP_data'


# ################# Run within release prediction experiments ###################
def run_within_release_prediction(prediction_model):
    threshold = 50
    wp_result_path = '%s/Result/WP/%s_%d/' % (root_path, getattr(prediction_model, '__name__'), threshold)
    make_path(wp_result_path)
    for project in get_project_list():
        within_release_prediction(project, 10, 10, prediction_model, threshold, wp_result_path)


# ################# Run cross release prediction experiments ###################
def run_cross_release_prediction(prediction_model):
    # thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    thresholds = [50]
    for threshold in thresholds:
        cp_result_path = '%s/Result/CP/%s_%d/' % (root_path, getattr(prediction_model, '__name__'), threshold)
        make_path(cp_result_path)
        for project, releases in get_project_releases_dict().items():
            cross_release_prediction(project, releases, prediction_model, threshold, cp_result_path)

        # 组合评估结果文件
        combine_results(cp_result_path)


if __name__ == '__main__':
    model = AccessModel
    # 运行版本内预测实验
    # run_within_release_prediction(model)
    # 运行版本间预测实验
    run_cross_release_prediction(model)
