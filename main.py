# -*- coding:utf-8 -*-

import src.models.natural as natural
from src.exps.cross_release import *
from src.models.access import AccessModel, NT_Model, NFC_Model
from src.models.explain import *

# 忽略警告信息
from src.models.static_analysis_tools import detect_bugs_by_checkstyle_from_each_single_file, PMDModel, CheckStyleModel

warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

# Global control variable
Predict = 'Predict'
Eval = 'Evaluate'


# ################# Run cross release predict experiments ###################
def run_cross_release_predict(prediction_model, depend_model=None):
    # thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    thresholds = [50]
    for threshold in thresholds:
        for project, releases in get_project_releases_dict().items():
            predict_cross_release(project, releases, prediction_model, depend_model, threshold)


# ################# Run cross release eval experiments ###################
def run_cross_release_eval(prediction_model, depend_model=AccessModel, depend=False):
    # thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    thresholds = [50]
    for threshold in thresholds:
        for project, releases in get_project_releases_dict().items():
            # depend = True 时, Access 结果 依赖baseline方法, = False 时, 为自身的预测结果
            eval_cross_release(project, releases, prediction_model, depend_model, threshold, depend)

        cp_result_path = f'{root_path}Result/CP/{getattr(prediction_model, "__name__")}_{threshold}/'
        combine_cross_results(cp_result_path)
        # eval_ifa(cp_result_path)
        # combine_cross_results_for_each_project(cp_result_path)


if __name__ == '__main__':
    # ============================= Access ====================================
    run_cross_release_predict(AccessModel)
    # run_cross_release_eval(AccessModel)

    # ======================= SAT-based approaches ============================
    # run_cross_release_predict(PMDModel)
    # run_cross_release_predict(CheckStyleModel)
    # run_cross_release_eval(PMDModel)
    # run_cross_release_eval(CheckStyleModel)

    # ======================= LM-based approaches =============================
    # run_cross_release_predict(natural.LM_2_Model)
    # run_cross_release_eval(natural.LM_2_Model)

    # ======================= MI-based approaches =============================
    # run_cross_release_predict(TMI_LR_Model)
    # run_cross_release_predict(TMI_MNB_Model)
    # run_cross_release_predict(TMI_SVM_Model)
    # run_cross_release_predict(TMI_DT_Model)
    # run_cross_release_predict(TMI_RF_Model)
    # run_cross_release_predict(LineDPModel)
    # run_cross_release_eval(TMI_LR_Model)
    # run_cross_release_eval(TMI_MNB_Model)
    # run_cross_release_eval(TMI_SVM_Model)
    # run_cross_release_eval(TMI_DT_Model)
    # run_cross_release_eval(TMI_RF_Model)
    # run_cross_release_eval(LineDPModel)

    # ======================= CM-based approaches =============================
    # run_cross_release_predict(NFC_Model)
    # run_cross_release_predict(NT_Model)
    # run_cross_release_eval(NFC_Model)
    # run_cross_release_eval(NT_Model)

    # ========================= Comparison ====================================
    # run_cross_release_predict(AccessModel, PMDModel)
    # run_cross_release_eval(AccessModel, PMDModel, True)

    # run_cross_release_predict(AccessModel, CheckStyleModel)
    # run_cross_release_eval(AccessModel, CheckStyleModel, True)

    # run_cross_release_predict(AccessModel, natural.LM_2_Model)
    # run_cross_release_eval(AccessModel, natural.LM_2_Model, True)

    # run_cross_release_predict(AccessModel, TMI_LR_Model)
    # run_cross_release_eval(AccessModel, TMI_LR_Model, True)

    # run_cross_release_predict(AccessModel, TMI_MNB_Model)
    # run_cross_release_eval(AccessModel, TMI_MNB_Model, True)

    # run_cross_release_predict(AccessModel, TMI_SVM_Model)
    # run_cross_release_eval(AccessModel, TMI_SVM_Model, True)

    # run_cross_release_predict(AccessModel, TMI_DT_Model)
    # run_cross_release_eval(AccessModel, TMI_DT_Model, True)

    # run_cross_release_predict(AccessModel, TMI_RF_Model)
    # run_cross_release_eval(AccessModel, TMI_RF_Model, True)

    # run_cross_release_predict(AccessModel, LineDP_Model)
    # run_cross_release_eval(AccessModel, LineDP_Model, True)

    # run_cross_release_predict(AccessModel, NFC_Model)
    # run_cross_release_eval(AccessModel, NFC_Model, True)

    # run_cross_release_predict(AccessModel, NT_Model)
    # run_cross_release_eval(AccessModel, NT_Model, True)

    # natural.run_lm()
    pass
