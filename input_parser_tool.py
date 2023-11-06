#!/usr/bin/env python
# -*- coding=utf-8 -*-
import yaml

INPUT_FILE_NAME = 'input.yaml'


def read_yaml_file(file_path):
    """
    读取yaml文件
    :param file_path:文件路径, string类型
    :return:返回读取结果, dict类型
    """
    try:
        # open方法打开直接读出来
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = f.read()
            # 将结果转换为字典类型
            result = yaml.load(cfg,Loader=yaml.FullLoader)
    except Exception as error:
        raise ValueError(f"Read yaml file error. File path: {file_path}. Error info:{str(error)}")
    return result

def load_params(input_file_path):
    job_input = read_yaml_file(input_file_path)
    params = []
    for pdic in job_input['CustomParameters']:
        for p_name, p_value in pdic.items():
            params.append((p_name, p_value['Value']))
    return params


if __name__ == '__main__':
    pass
