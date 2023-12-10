import numpy as np


def tenByOneKm(x):
    return np.piecewise(
        x,
        [
            np.logical_and(0 <= x, x <= 1000),
            np.logical_and(1000 < x, x <= 2000),
            np.logical_and(2000 < x, x <= 3000),
            np.logical_and(3000 < x, x <= 4000),
            np.logical_and(4000 < x, x <= 5000),
            np.logical_and(5000 < x, x <= 6000),
            np.logical_and(6000 < x, x <= 7000),
            np.logical_and(7000 < x, x <= 8000),
            np.logical_and(8000 < x, x <= 9000),
            np.logical_and(9000 < x, x <= 10000),
        ],
        [
            lambda _: 15,
            lambda _: -15,
            lambda _: 15,
            lambda _: -15,
            lambda _: 15,
            lambda _: -15,
            lambda _: 15,
            lambda _: -15,
            lambda _: 15,
            lambda _: -15,
        ],
    )


def testCourse(x):
    if x <= 100:
        return 15
    if x <= 200:
        return -20
    if x <= 300:
        return 20
    else:
        return 0
