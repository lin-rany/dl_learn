import logging
import os
from datetime import datetime
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为字符串
    time_string = current_time.strftime("%Y-%m-%d:%H")
    # 创建一个handler，用于写入日志文件
    directory = os.getcwd() + '/run_logs'
    if not os.path.exists(directory):
        os.makedirs(directory)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # debug_handler
    debug_file_name = directory + '/' + time_string + '_debug.log'
    debug_handler = logging.FileHandler(debug_file_name)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    # debug_handler
    critical_file_name = directory + '/' + time_string + '_critical.log'
    critical_handler = logging.FileHandler(critical_file_name)
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)
    logger.addHandler(critical_handler)
    # 在每次运行程序时，先写入两空行用于区分不同的程序运行日志
    def log_start(file_name):
        with open(file_name, 'a') as f:
            f.write('\n\n\n==========log start==========\n\n')
    log_start(critical_file_name)
    log_start(debug_file_name)
    return logger
if __name__ == '__main__':
    logger = get_logger()
    logger.info('Hello World--info')
    logger.debug('Hello World--debug')
    logger.warning('Hello World-waring')
    logger.critical('Hello World-critical')
