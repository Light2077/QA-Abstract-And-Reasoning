import numpy as np


def fil(*a, fn=""):
    for i in a:
        print(i)

    if fn is not "":
        print(fn)


if __name__ == "__main__":

    fil(1,2,3, fn="asdf")
