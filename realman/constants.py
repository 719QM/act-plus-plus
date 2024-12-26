# 跨文件共享的常量
import pathlib
import os

import numpy as np

# OPEN:1 CLOSED:0
# HAND_OPEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
HAND_OPEN = np.array([999, 999, 999, 999, 999, 999])
HAND_CLOSED = np.array([1, 1, 1, 1, 1, 1])
# SHADOW_HAND_UNNORMALIZE = lambda x: x * (HAND_OPEN - HAND_CLOSED) + HAND_CLOSED
# HAND_UNNORMALIZE = lambda x: [(open_val - closed_val) * x + closed_val
#                                      for open_val, closed_val
#                                      in zip(HAND_OPEN, HAND_CLOSED)]
def HAND_UNNORMALIZE(x):
    return [(open_val - closed_val) * x + closed_val
            for open_val, closed_val in zip(HAND_OPEN, HAND_CLOSED)]
# SHADOW_HAND_NORMALIZE = lambda x: (x - HAND_CLOSED) / (HAND_OPEN - HAND_CLOSED)
# SHADOW_HAND_VELOCITY_NORMALIZE = lambda x: x / (HAND_OPEN - HAND_CLOSED)
# SHADOW_HAND_NORMALIZE = lambda x: [(sum(abs(x_val - closed_val)) / sum(abs(open_val - closed_val)))
#                                    for x_val, closed_val, open_val
#                                    in zip(x, HAND_CLOSED, HAND_OPEN)]
# SHADOW_HAND_VELOCITY_NORMALIZE = lambda x: [(sum(abs(x_val)) / sum(abs(open_val - closed_val)))
#                                    for x_val, closed_val, open_val
#                                    in zip(x, HAND_CLOSED, HAND_OPEN)]


def HAND_NORMALIZE(x):
    diff_closed_open = HAND_OPEN - HAND_CLOSED
    diff_x_closed = x - HAND_CLOSED
    sum_abs_diff_closed_open = np.sum(np.abs(diff_closed_open))
    sum_abs_diff_x_closed = np.sum(np.abs(diff_x_closed))

    # 避免除以零
    if sum_abs_diff_closed_open == 0:
        return 0
    else:
        return sum_abs_diff_x_closed / sum_abs_diff_closed_open

def HAND_VELOCITY_NORMALIZE(x):
    diff_closed_open = HAND_OPEN - HAND_CLOSED
    sum_abs_diff_closed_open = np.sum(np.abs(diff_closed_open))
    sum_abs_diff_x = np.sum(np.abs(x))

    # 避免除以零
    if sum_abs_diff_closed_open == 0:
        return 0
    else:
        return sum_abs_diff_x / sum_abs_diff_closed_open