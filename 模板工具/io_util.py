# 文件读写工具包装
import json,os,errno

def try_dirs(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def write_json_to_file(filepath, data):
    """写入JSON数据到给定文件路径中,并覆盖原文件。

    参数：
        filepath (str): 文件路径
        data (dict): 包含新数据的字典

    返回:
        无
    """
    try_dirs(filepath)
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file,ensure_ascii=False)

def read_json_from_file(filepath):
    read_ok=False
    results={}
    if os.path.exists(filepath):
        read_ok=True
        with open(filepath, 'r') as f:
            results = json.load(f)
    return read_ok,results

def write_lines_to_file(filename, lines):
    try_dirs(filename)
    with open(filename, 'w') as file:
        for line in lines:
            file.write(line + '\n')