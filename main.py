# -*- coding:utf-8 -*-
import warnings
import sys
from src.models.tools import *
from src.models.explain import *
from src.models.glance import *
from src.models.natural import *

# 忽略警告信息
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

# The model name and its corresponding python class implementation
MODEL_DICT = {'TMI-LR': TMI_LR, 'TMI-SVM': TMI_SVM, 'TMI-MNB': TMI_MNB, 'TMI-DT': TMI_DT, 'TMI-RF': TMI_RF,
              'LineDP': LineDP,
              'PMD': PMD, 'CheckStyle': CheckStyle,
              'NGram': NGram, 'NGram-C': NGram_C,
              'Glance-EA': Glance_EA, 'Glance-MD': Glance_MD, 'Glance-LR': Glance_LR,
              }


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


def run_default():
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
    # run_cross_release_predict(Glance_Resort)
    run_cross_release_predict(Glance_LR)


def parse_args():
    # 没有参数列表,执行当前程序中定义的方法
    if len(sys.argv) == 1:
        run_default()
    else:
        model_name = sys.argv[1]
        run_cross_release_predict(MODEL_DICT[model_name])


if __name__ == '__main__':
    parse_args()
