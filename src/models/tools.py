# -*- coding:utf-8 -*-

import os

from src.utils.eval import evaluation
from src.utils.helper import *
import xml.dom.minidom as minidom


def get_version_info(project, release):
    """
    Get version information. commit id, version name, version date, next date, branch. OK.
    :return:
    """
    commit_version_code, commit_version_branch = {}, {}
    lines = read_data_from_file(f'{root_path}DataCollection/Version/{project}.csv')
    for line in lines[1:]:
        spices = line.strip().split(",")
        version_code = spices[0]
        version_name = spices[1]
        version_branch = spices[4]
        commit_version_code[version_name] = version_code
        commit_version_branch[version_name] = version_branch

    return commit_version_code[release], commit_version_branch[release]


# ############################################## FindBugs ########################################################
def detect_bugs_by_findbugs(project, release):
    result_file = f'{root_path}Archive/{project}/fb_{release}.xml'

    target_jar = f'{root_path}Archive/{project}/{release}.jar'
    # aux_classes = f'{root_path}Archive/{project}/auxclasses/'

    cmd_findbugs = f'findbugs -textui -low -xml -outputFile fb_{result_file} {target_jar}'
    os.system(cmd_findbugs)


# ################################################ PMD ##########################################################

def detect_bugs_by_pmd():
    """
    使用PMD检测bug OK
    :return:
    """
    for project, releases in get_project_releases_dict().items():
        for release in releases:
            # 检测某一个项目的具体版本
            print(f'{"=" * 30} Detecting bugs of {release} by PMD tool {"=" * 30}')

            project_result_details_path = f'{result_path}PMD/detailed_result/{project}'
            source_code_path = f'{root_path}Repository/{project}/'

            make_path(project_result_details_path)

            # 切换到准备待检测的代码
            os.chdir(source_code_path)
            version_code, version_branch = get_version_info(project, release)
            os.system(f'git checkout -f {version_branch}')
            os.system(f'git reset --hard {version_code}')

            # 需要检测的规则
            rule_list = [
                'category/java/bestpractices.xml',
                'category/java/codestyle.xml',
                'category/java/design.xml',
                'category/java/errorprone.xml',
                'category/java/multithreading.xml',
                'category/java/performance.xml',
                'category/java/security.xml',
            ]

            output_file = f'{project_result_details_path}/pmd_{release}.csv'
            cmd_pmd = f'pmd -d {source_code_path} -R {",".join(rule_list)} -f csv > {output_file}'
            os.system(cmd_pmd)


def parse_detailed_result_of_pmd():
    """
    解析PMD的结果文件 OK
    :return:
    """
    for project, releases in get_project_releases_dict().items():
        for release in releases:
            # 转换某一个项目的具体版本的结果
            print(f'{"=" * 30} Parse bugs of {release} by PMD tool {"=" * 30}')

            project_result_root_path = f'{result_path}PMD/final_result/{project}'
            output_file = f'{result_path}PMD/detailed_result/{project}/pmd_{release}.csv'
            source_code_path = f'{root_path}Repository/{project}/'
            make_path(project_result_root_path)

            file_buggy_dict = {}
            lines = read_data_from_file(output_file)
            for line in lines[1:]:
                split = line.split(',')
                file_name = split[2].strip('"').replace("\\", "/").replace(source_code_path, "")
                line_number = split[4].strip('"')
                priority = int(split[3].strip('"'))
                if file_name not in file_buggy_dict:
                    file_buggy_dict[file_name] = [[line_number, priority]]
                elif [line_number, priority] not in file_buggy_dict[file_name]:
                    file_buggy_dict[file_name].append([line_number, priority])

            text = ''
            for file_name, buggy_list in file_buggy_dict.items():
                p_set = sorted(to_set([buggy[1] for buggy in buggy_list]))
                lines = []
                for p in p_set:
                    lines += [f'{buggy[0]}:{buggy[1]}' for buggy in buggy_list if buggy[1] == p]
                text += f'{file_name},{",".join(lines)}\n'

            save_csv_result(f'{project_result_root_path}/pmd_{release}.csv', text)


# ############################################# CheckStyle #####################################################

def detect_bugs_by_checkstyle():
    """
    使用CheckStyle检测bug
    :return:
    """
    for project, releases in get_project_releases_dict().items():
        for release in releases:
            # 检测某一个项目的具体版本
            print(f'{"=" * 30} Detecting bugs of {release} by CheckStyle tool {"=" * 30}')

            project_result_details_path = f'{result_path}CheckStyle/detailed_result/{project}'
            source_code_path = f'{root_path}Repository/{project}/'

            make_path(project_result_details_path)

            # 切换到准备待检测的代码
            os.chdir(source_code_path)
            version_code, version_branch = get_version_info(project, release)
            os.system(f'git checkout -f {version_branch}')
            os.system(f'git reset --hard {version_code}')

            # 需要检测的规则
            rule_list = ['sun', 'google']

            for check in rule_list:
                main_class = 'com.puppycrawl.tools.checkstyle.Main'
                output_file = f'{project_result_details_path}/cs_{check}_{release}.xml'
                config_file = f'%CHECKSTYLE_HOME%/{check}_checks.xml'  # docs/google_checks.xml
                cmd_cs = f'java {main_class} -c {config_file} -f xml -o {output_file} {source_code_path}'
                os.system(cmd_cs)


def is_legal_file(path):
    if not os.path.exists(path):
        return False
    lines = read_data_from_file(path)
    return len(lines) > 0 and lines[-1].strip() == '</checkstyle>'


def parse_file_xml_get_buggy_lines(file_path):
    # 读取原始的bug报告
    DOMTree = minidom.parse(file_path)
    body = DOMTree.documentElement.getElementsByTagName('error')
    buggy_lines = [bug.getAttribute('line') for bug in body]
    return buggy_lines


def to_set(buggy_list):
    buggy_set = []
    for buggy in buggy_list:
        if buggy not in buggy_set:
            buggy_set.append(buggy)
    return buggy_set


def parse_project_xml(file_path):
    # 读取原始的bug报告
    DOMTree = minidom.parse(file_path)
    files = DOMTree.documentElement.getElementsByTagName('file')

    file_buggy_dict = {}
    for file in files:
        file_name = file.getAttribute('name')
        buggy_lines = [bug.getAttribute('line') for bug in file.getElementsByTagName('error')]
        file_buggy_dict[file_name] = buggy_lines
    return file_buggy_dict


def combine_sun_and_google(path, sun_dict, google_dict):
    file_names = sun_dict.keys()
    for name_in_google in google_dict.keys():
        if name_in_google not in file_names:
            file_names.append(name_in_google)
    text = ''
    for file_name in file_names:
        lines = []
        if file_name in sun_dict:
            lines += [f'{line_number}:1' for line_number in sun_dict[file_name]]
        if file_name in google_dict:
            lines += [f'{line_number}:2' for line_number in google_dict[file_name]]
        file_name = file_name.replace("\\", "/").replace(path, "")
        text += f'{file_name},{",".join(lines)}\n'
    return text


def parse_detailed_result_of_checkstyle():
    error_project_release_list = []
    for proj, releases in get_project_releases_dict().items():
        for rel in releases:
            # print(f'{"=" * 30} Parse bugs of {rel} by CheckStyle tool {"=" * 30}')
            prefix_path = f'{root_path}Repository/{proj}/'

            output_file_path = f'{result_path}CheckStyle/final_result/{proj}/'
            make_path(output_file_path)
            method_root_path = f'{result_path}CheckStyle/detailed_result/{proj}/cs_sun_{rel}.csv'
            if not is_legal_file(method_root_path):
                error_project_release_list.append([proj, rel])
                print(f'Error sun  {rel}  -------------------- !')
                continue
            sun_dict = parse_project_xml(method_root_path)

            method_root_path = f'{result_path}CheckStyle/detailed_result/{proj}/cs_google_{rel}.csv'
            if not is_legal_file(method_root_path):
                if [proj, rel] not in error_project_release_list:
                    error_project_release_list.append([proj, rel])
                print(f'Error google {rel}  -------------------- !')
                continue
            google_dict = parse_project_xml(method_root_path)

            text = combine_sun_and_google(prefix_path, sun_dict, google_dict)
            save_csv_result(f'{output_file_path}cs_{rel}.csv', text)
    # 处理出错的项目
    for item in error_project_release_list:
        detect_bugs_by_checkstyle_from_each_single_file(item[0], item[1])


def detect_bugs_by_checkstyle_from_each_single_file(project, release):
    """
    当解析整个文件失败的时候调用这个函数
    :param project:
    :param release:
    :return:
    """
    print(f'{"=" * 30} Detecting bugs of {release} by CheckStyle tool {"=" * 30}')

    # File path
    detail_result_path = f'{result_path}CheckStyle/detailed_result/{project}/tmp-{release}/'
    final_result_path = f'{result_path}CheckStyle/final_result/{project}/'
    make_path(detail_result_path)
    make_path(final_result_path)

    source_root_path = f'{root_path}Repository/{project}/'
    os.chdir(source_root_path)
    version_code, version_branch = get_version_info(project, release)
    os.system(f'git checkout -f {version_branch}')
    os.system(f'git reset --hard {version_code}')

    text = ''
    source_files = export_all_files_in_project(source_root_path)
    for index in range(len(source_files)):
        source_file = source_files[index]
        print(f'Processing {index}/{len(source_files)} {source_file}')
        buggy_lines = []
        p = 1
        for check in ['sun', 'google']:
            config_file = f'%CHECKSTYLE_HOME%/{check}_checks.xml'  # docs/google_checks.xml
            detail_file_path = detail_result_path + source_file.replace('/', '.') + '.xml'
            source_file_path = source_root_path + source_file
            # command
            main_class = 'com.puppycrawl.tools.checkstyle.Main'
            cmd_cs = f'java {main_class} -c {config_file} -f xml -o {detail_file_path} {source_file_path}'
            os.system(cmd_cs)
            if not is_legal_file(detail_file_path):
                continue

            buggy_lines += [f'{line}:{p}' for line in to_set(parse_file_xml_get_buggy_lines(detail_file_path))]
            p += 1
        text += f'{source_file},{",".join(buggy_lines)}\n' if len(buggy_lines) != 0 else ''

    save_csv_result(f'{final_result_path}cs_{release}.csv', text)


# ############################################# CheckStyle #####################################################

def read_SAT_result(path):
    file_line_weight_dict = {}
    for file in read_data_from_file(path):
        split = file.strip().split(',')
        temp_dict = {}
        for line in split[1:]:
            s = line.split(':')
            if len(s) < 2:
                continue
            line_number, weight = int(s[0]), 1 / int(s[1])
            temp_dict[line_number] = weight
        file_line_weight_dict[split[0]] = temp_dict
    return file_line_weight_dict


def PMDModel(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    file_line_weight_dict = read_SAT_result(f'{result_path}PMD/final_result/{proj.split("-")[0]}/pmd_{proj}.csv')

    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}
    defect_cf_dict = {}
    effort_cf_dict = {}

    # 预测值为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])

    # 对预测为有bug的文件逐个进行代码行级别的排序
    for i in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[i]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的测试文件(被预测为有bug,但实际上)没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ############################ 重点,怎么给每行赋一个缺陷值 ################################
        # 计算 每一行的权重, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数
        hit_count = np.array([.0] * len(target_file_lines))
        if target_file_name in file_line_weight_dict:
            for index in range(len(target_file_lines)):
                if index + 1 in file_line_weight_dict[target_file_name]:
                    hit_count[index] = file_line_weight_dict[target_file_name][index + 1]

        # ############################ 重点,怎么给每行赋一个缺陷值 ################################

        # 根据命中次数对代码行进行降序排序, 按照排序后数值从大到小的顺序显示代码行在原列表中的索引, cut_off 为切分点
        # line + 1,因为下标是从0开始计数而不是从1开始
        sorted_index = np.argsort(-hit_count)
        sorted_line_number = [line + 1 for line in sorted_index.tolist()]
        # 原始未经过调整的列表
        ranked_list_dict[target_file_name] = sorted_line_number

        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################
        # 需要调整为最差排序的列表,当分数相同时
        worst_line_number = list(sorted_line_number)
        sorted_list = hit_count[sorted_index]
        worse_list, current_score, start_index, oracle_lines = [], -1, -1, oracle_line_dict[target_file_name]
        for ii in range(len(sorted_list)):
            if sorted_list[ii] != current_score:
                current_score = sorted_list[ii]
                start_index = ii
            elif worst_line_number[ii] not in oracle_lines:
                temp = worst_line_number[ii]  # 取出这个无bug的行号
                for t in range(ii, start_index, -1):
                    worst_line_number[t] = worst_line_number[t - 1]
                worst_line_number[start_index] = temp
        worst_list_dict[target_file_name] = worst_line_number
        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################

        # ###################################### 切分点设置 ################################################
        # 20% effort (i.e., LOC)
        effort_cf_dict[target_file_name] = int(.2 * len(target_file_lines))
        # 设置分类切分点: 默认前50%
        defect_cf_dict[target_file_name] = int(threshold / 100.0 * len(target_file_lines))
        # defect_cf_dict[target_file_name] = len([hit for hit in hit_count if hit > 0])

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)


def CheckStyleModel(proj, vector, classifier, test_text_lines, test_filename, test_predictions, out_file, threshold):
    file_line_weight_dict = read_SAT_result(f'{result_path}CheckStyle/final_result/{proj.split("-")[0]}/cs_{proj}.csv')

    # 正确bug行号字典 预测bug行号字典 二分类切分点字典 工作量切分点字典
    oracle_line_dict = read_line_level_dataset(proj)
    ranked_list_dict = {}
    worst_list_dict = {}
    defect_cf_dict = {}
    effort_cf_dict = {}

    # 预测值为有缺陷的文件的索引
    defect_prone_file_indices = np.array([index[0] for index in np.argwhere(test_predictions == 1)])

    # 对预测为有bug的文件逐个进行代码行级别的排序
    for i in range(len(defect_prone_file_indices)):
        target_file_index = defect_prone_file_indices[i]
        # 目标文件名
        target_file_name = test_filename[target_file_index]
        # 有的测试文件(被预测为有bug,但实际上)没有bug,因此不会出现在 oracle 中,这类文件要剔除
        if target_file_name not in oracle_line_dict:
            oracle_line_dict[target_file_name] = []
        # 目标文件的代码行列表
        target_file_lines = test_text_lines[target_file_index]

        # ############################ 重点,怎么给每行赋一个缺陷值 ################################
        # 计算 每一行的权重, 初始为 [0 0 0 0 0 0 ... 0 0], 注意行号从0开始计数
        hit_count = np.array([.0] * len(target_file_lines))
        if target_file_name in file_line_weight_dict:
            for index in range(len(target_file_lines)):
                if index + 1 in file_line_weight_dict[target_file_name]:
                    hit_count[index] = file_line_weight_dict[target_file_name][index + 1]

        # ############################ 重点,怎么给每行赋一个缺陷值 ################################

        # 根据命中次数对代码行进行降序排序, 按照排序后数值从大到小的顺序显示代码行在原列表中的索引, cut_off 为切分点
        # line + 1,因为下标是从0开始计数而不是从1开始
        sorted_index = np.argsort(-hit_count)
        sorted_line_number = [line + 1 for line in sorted_index.tolist()]
        # 原始未经过调整的列表
        ranked_list_dict[target_file_name] = sorted_line_number

        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################
        # 需要调整为最差排序的列表,当分数相同时
        worst_line_number = list(sorted_line_number)
        sorted_list = hit_count[sorted_index]
        worse_list, current_score, start_index, oracle_lines = [], -1, -1, oracle_line_dict[target_file_name]
        for ii in range(len(sorted_list)):
            if sorted_list[ii] != current_score:
                current_score = sorted_list[ii]
                start_index = ii
            elif worst_line_number[ii] not in oracle_lines:
                temp = worst_line_number[ii]  # 取出这个无bug的行号
                for t in range(ii, start_index, -1):
                    worst_line_number[t] = worst_line_number[t - 1]
                worst_line_number[start_index] = temp
        worst_list_dict[target_file_name] = worst_line_number
        # ####################################### 将序列调整为理论上的最差性能  实际使用时可以去掉 ################

        # ###################################### 切分点设置 ################################################
        # 20% effort (i.e., LOC)
        effort_cf_dict[target_file_name] = int(.2 * len(target_file_lines))
        # 设置分类切分点: 默认前50%
        # defect_cf_dict[target_file_name] = len([hit for hit in hit_count if hit > 0])
        defect_cf_dict[target_file_name] = int(threshold / 100.0 * len(target_file_lines))

    dump_pk_result(out_file, [oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict])
    return evaluation(proj, oracle_line_dict, ranked_list_dict, worst_list_dict, defect_cf_dict, effort_cf_dict)


def run_hint():
    # run pmd
    # detect_bugs_by_pmd()
    # parse_detailed_result_of_pmd()

    # run checkstyle
    # detect_bugs_by_checkstyle()
    # parse_detailed_result_of_checkstyle()
    # detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.2.0')
    pass


if __name__ == '__main__':
    # parse_detailed_result_of_pmd()
    parse_detailed_result_of_checkstyle()
    pass
