# -*- coding:utf-8 -*-

import os
import shutil
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
                    lines += [buggy[0] for buggy in buggy_list if buggy[1] == p]
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
            lines += sun_dict[file_name]
        if file_name in google_dict:
            lines += google_dict[file_name]
        file_name = file_name.replace("\\", "/").replace(path, "")
        text += f'{file_name},{",".join(lines)}\n'
    return text


def parse_detailed_result_of_checkstyle():
    error_project_release_list = []
    for proj, releases in get_project_releases_dict().items():
        for rel in releases:
            prefix_path = f'{root_path}Repository/{proj}/'
            # print(rel)
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

            buggy_lines += to_set(parse_file_xml_get_buggy_lines(detail_file_path))

        text += f'{source_file},{",".join(buggy_lines)}\n' if len(buggy_lines) != 0 else ''

    save_csv_result(f'{final_result_path}cs_{release}.csv', text)


if __name__ == '__main__':
    # run pmd
    # detect_bugs_by_pmd()
    # parse_detailed_result_of_pmd()

    # run checkstyle
    # detect_bugs_by_checkstyle()
    # parse_detailed_result_of_checkstyle()
    # detect_bugs_by_checkstyle_from_each_single_file('ww', 'ww-2.2.0')
    pass
