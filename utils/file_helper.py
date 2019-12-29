from utils.config import *
import time

def get_result_file_name():
    now = time.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = os.path.join(RESULT_DIR, now+"_result.csv")
    return file_name