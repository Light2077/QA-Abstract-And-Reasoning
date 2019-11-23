from multiprocessing import Pool, cpu_count
import numpy as np
import time
"""
主要展示多进程计算的好处，多进程的模板参见mul_cal()这个函数
需要数据分发，数据汇总两个步骤

"""

def cal(arr):
    sum1 = 0
    for i in arr:

        # 这一部分模拟浪费时间
        waste = 0
        for j in range(3000000):
            waste += j
        sum1 += i

    res = sum1
    return res

def mul_cal(arr):
    core = 12
    with Pool(core) as p:
        arr_sp = np.array_split(arr, core)
        res = sum(p.map(cal, arr_sp))
    return res

def gen_num():
    for i in range(10000):
        yield i

if __name__ == '__main__':
    print("cpu核心数：{}".format(cpu_count()))
    arr = np.arange(1, 500, 1)
    t0 = time.time()
    #res1 = cal(arr)
    t1 = time.time()
    print("单进程共耗时：{:.2f}s".format(t1 - t0))

    print("上下切分")

    t0 = time.time()
    res2 = mul_cal(arr)
    t1 = time.time()
    print("多进程共耗时：{:.2f}s".format(t1 - t0))