# -*- coding:utf-8 -*-

import src.models.natural as natural
from src.exps.cross_release import *
from src.models.access import AccessModel
from src.models.explain import *

# 忽略警告信息
from src.models.static_analysis_tools import detect_bugs_by_checkstyle_from_each_single_file, PMDModel, CheckStyleModel

warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

# Global control variable
Predict = 'Predict'
Eval = 'Evaluate'


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
                depend_model = LineDPModel
                eval_cross_release(project, releases, cp_result_path, depend_model, depend=False)
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
    # run_cross_release_prediction(TMI_MNB_Model, Predict)

    # run_cross_release_prediction(TMI_RF_Model, Predict)
    # run_cross_release_prediction(TMI_Tree_Model, Predict)

    natural.run_lm()
    pass
