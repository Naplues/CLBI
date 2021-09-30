import numpy as np
import pandas as pd
import time, math, os, re, pickle
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, \
    classification_report

# data_dir = './data/'

python_common_tokens = ['abs', 'delattr', 'hash', 'memoryview', 'set', 'all', 'dict', 'help', 'min', 'setattr', 'any',
                        'dir', 'hex', 'next', 'slice', 'ascii', 'divmod', 'id', 'object', 'sorted', 'bin', 'enumerate',
                        'input', 'oct', 'staticmethod', 'bool', 'eval', 'int', 'open', 'str', 'breakpoint', 'exec',
                        'isinstance', 'ord', 'sum', 'bytearray', 'filter', 'issubclass', 'pow', 'super', 'bytes',
                        'float', 'iter', 'print', 'tuple', 'callable', 'format', 'len', 'property', 'type', 'chr',
                        'frozenset', 'list', 'range', 'vars', 'classmethod', 'getattr', 'locals', 'repr', 'zip',
                        'compile', 'globals', 'map', 'reversed', '__import__', 'complex', 'hasattr', 'max', 'round',
                        'False', 'await', 'else', 'import', 'passNone', 'break', 'except', 'in', 'raise', 'True',
                        'class', 'finally', 'is', 'return', 'and', 'continue', 'for', 'lambda', 'try', 'as', 'def',
                        'from', 'nonlocal', 'while', 'assert', 'del', 'global', 'not', 'with', 'async', 'elif', 'if',
                        'or', 'yield', 'self']


def preprocess_code_line(code, remove_python_common_tokens=False):
    """
    OK
    对代码进行预处理
    :param code:
    :param remove_python_common_tokens:
    :return:
    """
    code = code.replace('(', ' ') \
        .replace(')', ' ') \
        .replace('{', ' ') \
        .replace('}', ' ') \
        .replace('[', ' ') \
        .replace(']', ' ') \
        .replace('.', ' ') \
        .replace(':', ' ') \
        .replace(';', ' ') \
        .replace(',', ' ') \
        .replace(' _ ', '_')
    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)

    processed_code = code
    if remove_python_common_tokens:
        new_code = ''
        for tok in code.split():
            if tok not in python_common_tokens:
                new_code = new_code + tok + ' '
        processed_code = new_code.strip()
    return processed_code.strip()


def load_data(proj, mode='train', use_text=True, remove_python_common_tokens=False, data_dir='./data/'):
    if mode == 'train':
        data = pickle.load(open(data_dir + proj + '_train.pkl', 'rb'))
    elif mode == 'test':
        data = pickle.load(open(data_dir + proj + '_test.pkl', 'rb'))
    else:
        print('input mode is wrong')
        return

    dict = pickle.load(open(data_dir + proj + '_dict.pkl', 'rb'))[1]
    # print(dict[1])
    max_idx = np.max(list(dict.values()))
    # print(max_idx)
    dict['<STR>'] = max_idx + 1
    # print(dict)
    commit_id, label, all_code_change = data[0], data[1], data[3]

    if use_text:
        all_added_code, all_removed_code = [], []
        for code_change in all_code_change:
            added_code_list, removed_code_list = [], []

            for i in range(0, len(code_change)):
                ch = code_change[i]
                added_code = ch['added_code']
                removed_code = ch['removed_code']

                if len(added_code) > 0:
                    for code in added_code:
                        if code.startswith("#"):
                            continue
                        #    print('code is')
                        #    print(code)
                        added_code_list.append(preprocess_code_line(code, remove_python_common_tokens))

                if len(removed_code) > 0:
                    for code in removed_code:
                        if code.startswith("#"):
                            continue
                        removed_code_list.append(preprocess_code_line(code, remove_python_common_tokens))

            #         added_code_list = list(set(added_code_list))
            #         removed_code_list = list(set(removed_code_list))

            all_added_code.append(added_code_list)
            all_removed_code.append(removed_code_list)

        all_added_code = [' \n '.join(list(set(code))) for code in all_added_code]
        all_removed_code = [' \n '.join(list(set(code))) for code in all_removed_code]

        return all_added_code, all_removed_code, commit_id, dict, label
    else:
        return commit_id, label


def load_change_metrics_df(cur_proj):
    data_path = './data/'
    change_metrics = pd.read_csv(data_path + cur_proj + '_metrics.csv')
    change_metrics = change_metrics.drop(
        ['author_date', 'bugcount', 'fixcount', 'revd', 'tcmt', 'oexp', 'orexp', 'osexp', 'osawr'], axis=1)
    change_metrics = change_metrics.fillna(value=0)
    return change_metrics


# count_vect: fitted CountVectorizer()
def combine_features(combined_code, change_metrics, count_vect, commit_id, label, use_text_feature=True,
                     use_change_metrics=True):
    if use_text_feature == False and use_change_metrics == False:
        return

    if use_text_feature and use_change_metrics:
        tmp_df = pd.DataFrame()
        tmp_df['code_change'] = combined_code
        tmp_df['commit_id'] = commit_id
        tmp_df['label'] = label

        tmp_features_df = tmp_df.merge(change_metrics, left_on='commit_id', right_on='commit_id')

        #         print('merge 2 features df complete')

        new_label = tmp_features_df['label']
        new_commit_id = tmp_features_df['commit_id']

        code_change = list(tmp_features_df['code_change'])
        code_change_arr = count_vect.transform(code_change)
        code_change_arr = code_change_arr.astype(np.int8).toarray()

        features_df = pd.DataFrame(code_change_arr, columns=count_vect.get_feature_names())
        features_df = features_df.astype(np.int8)

        for metrics in tmp_features_df.columns[3:]:
            features_df[metrics] = tmp_features_df[metrics]
            features_df[metrics] = features_df[metrics].astype(np.float32)

        del tmp_features_df, tmp_df, code_change, code_change_arr
        #         del tmp_df
        #         del code_change
        #         del code_change_arr

        #         code_change_arr = count_vect.transform(combined_code)
        #         code_change_arr = code_change_arr.astype(np.int16).toarray()
        #         code_change_df = pd.DataFrame(code_change_arr, columns=count_vect.get_feature_names())
        #         code_change_df['commit_hash'] = commit_id
        #         code_change_df['label'] = label
        #         features_df = code_change_df.merge(change_metrics,left_on='commit_hash',right_on='commit_id')

        #         new_label = features_df['label']
        #         new_commit_id = features_df['commit_hash']

        #         features_df = features_df.drop(['commit_id','commit_hash','label'],axis=1)

        return features_df, new_commit_id, new_label

    if use_text_feature and not use_change_metrics:
        code_change_arr = count_vect.transform(combined_code).astype(np.int16).toarray()
        code_change_df = pd.DataFrame(code_change_arr, columns=count_vect.get_feature_names())

        return code_change_df, commit_id, label

    if not use_text_feature and use_change_metrics:
        features_df = pd.DataFrame()
        features_df['commit_hash'] = commit_id
        features_df['label'] = label
        features_df = features_df.merge(change_metrics, left_on='commit_hash', right_on='commit_id')
        new_label = features_df['label']

        features_df = features_df.drop(['commit_id', 'commit_hash', 'label'], axis=1)
        #         features_df = change_metrics[change_metrics['commit_id'].isin(commit_id)]
        #         features_df = features_df.drop(['commit_id'],axis=1)
        return features_df, new_label


def prepare_data(cur_proj, mode='train', use_text=True, remove_python_common_tokens=False, data_dir='./data/'):
    if use_text:
        all_added_code, all_removed_code, commit_id, dict, label = load_data(cur_proj, mode=mode, use_text=use_text,
                                                                             remove_python_common_tokens=remove_python_common_tokens,
                                                                             data_dir=data_dir)
        combined_code = []

        for i in range(0, len(all_added_code)):
            combined_code.append(all_added_code[i] + ' ' + all_removed_code[i])

        return combined_code, commit_id, label
    else:
        commit_id, label = load_data(cur_proj, mode=mode, use_text=use_text)
        return commit_id, label


def train_eval_model(clf, x_train, y_train, x_test, y_test):
    start = time.time()
    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_test)[:, 1]

    pred = clf.predict(x_test)

    pred_df = pd.DataFrame()
    pred_df['prob'] = prob
    pred_df['pred'] = pred
    pred_df['actual'] = y_test

    return clf, pred_df


def create_path_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
