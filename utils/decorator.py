import time
from functools import wraps

def count_time(func):
    @wraps(func)
    def int_time(*args, **kwargs):

        start_time = time.time()  # 程序开始时间
        res = func(*args, **kwargs)
        over_time = time.time()  # 程序结束时间

        total_time = (over_time - start_time)
        print('程序{}()共耗时{:.2f}秒'.format(func.__name__, total_time))
        return res

    return int_time