# -*- coding:utf-8 -*-
import warnings

from src.models.tools import *
from src.models.explain import *
from src.models.glance import *
from src.models.natural import *

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)


# ========================= Run RQ1 experiments =================================
def run_cross_release_predict(prediction_model):
    for project, releases in get_project_releases_dict().items():
        print(f'========== {prediction_model.model_name} CR PREDICTION for {project} ========================'[:60])
        for i in range(len(releases) - 1):
            # 1. Loading data. train data index = i, test data index = i + 1
            model = prediction_model(releases[i], releases[i + 1])

            model.file_level_prediction()
            model.analyze_file_level_result()

            model.line_level_prediction()
            model.analyze_line_level_result()


if __name__ == '__main__':
    # ======================= MI-based approaches =============================
    # run_cross_release_predict(TMI_LR)
    # run_cross_release_predict(TMI_SVM)
    # run_cross_release_predict(TMI_MNB)
    # run_cross_release_predict(TMI_DT)
    # run_cross_release_predict(TMI_RF)
    # run_cross_release_predict(LineDP)

    # ======================= SAT-based approaches ============================
    # run_cross_release_predict(PMD)
    # run_cross_release_predict(CheckStyle)

    # ======================= LM-based approaches =============================
    # run_cross_release_predict(NGram)
    # run_cross_release_predict(NGram_C)

    # ======================= CM-based approaches =============================
    # run_cross_release_predict(Glance_EA)
    # run_cross_release_predict(Glance_MD)
    run_cross_release_predict(Glance2)

    pass
