import numpy as np


def testCourse(x):
    if x <= 100:
        return 15
    if x <= 200:
        return -20
    if x <= 300:
        return 20
    else:
        return 0
