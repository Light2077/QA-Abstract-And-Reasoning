from utils.config import *
import time

def get_result_file_name():
    """
    获取结果文件的名称
    :return:  20191209_16h49m32s_res.csv
    """
    now = time.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = os.path.join(RESULT_DIR, now+"_result.csv")
    return file_name