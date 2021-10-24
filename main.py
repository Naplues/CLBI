# -*- coding:utf-8 -*-

import src.models.natural as natural
from src.exps.cross_release import *
from src.models.glance import Glance
from src.models.explain import *

# 忽略警告信息
from src.models.tools import detect_bugs_by_checkstyle_from_each_single_file, PMDModel, CheckStyleModel

warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

# Global control variable
Predict = 'Predict'
Eval = 'Evaluate'


# thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


# ################# Run CR-P experiments ###################
def run_cross_release_predict(prediction_model, depend_model=None):
    for project, releases in get_project_releases_dict().items():
        print(f'========== Cross-release prediction for {project} ========================================'[:50])
        # 某个预测模型的跨版本实验结果存储路径. 如, D:/CLDP_data/Result/CP/AccessModel_50/

        for i in range(len(releases) - 1):
            # 1. Loading data. train data index = i, test data index = i + 1
            train_release, test_release = releases[i], releases[i + 1]

            model = prediction_model(train_release, test_release)

            model.file_level_prediction()
            model.analyze_file_level_result()

            model.line_level_prediction()
            model.analyze_line_level_result()


# ################# Run CR-E experiments ###################
def run_cross_release_eval(prediction_model, depend=False):
    for project, releases in get_project_releases_dict().items():
        print(f'========== Cross-release prediction for {project} ========================================'[:50])
        # 某个预测模型的跨版本实验结果存储路径. 如, D:/CLDP_data/Result/CP/AccessModel_50/
        print("Training set\t ===> \tTest set.")
        for i in range(len(releases) - 1):
            # 1. Loading data. train data index = i, test data index = i + 1
            train_release, test_release = releases[i], releases[i + 1]

            model = prediction_model(train_release, test_release)
            model.analyze_file_level_result()
            model.analyze_line_level_result()


if __name__ == '__main__':
    # run_cross_release_predict(LineDP)
    run_cross_release_predict(natural.Entropy)

    # ============================= Access ====================================
    # run_cross_release_predict(AccessModel)
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
    # run_cross_release_predict(LineDP_Model)
    # run_cross_release_eval(TMI_LR_Model)
    # run_cross_release_eval(TMI_MNB_Model)
    # run_cross_release_eval(TMI_SVM_Model)
    # run_cross_release_eval(TMI_DT_Model)
    # run_cross_release_eval(TMI_RF_Model)
    # run_cross_release_eval(LineDP_Model)

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
    # run_cross_release_predict(LineDP_Model)

    pass
