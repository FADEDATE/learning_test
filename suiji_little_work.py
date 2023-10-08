import numpy as np
import numpy.linalg as lng
import math
import pandas as pd
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint

import matplotlib.pyplot as plt

np.set_printoptions(precision=6)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 步骤一（替换sans-serif字体）
plt.rcParams["axes.unicode_minus"] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 参考：https://www.cnblogs.com/jjmg/p/grey_model_by_python.html
# 其他案例：https://github.com/dontLoveBugs/GM-1-1


# 线性平移预处理，确保数据级比在可容覆盖范围
def greyModelPreprocess(dataVec):
    "Set linear-bias c for dataVec"

    c = 0
    x0 = np.array(dataVec, float)
    n = x0.shape[0]  # 行数

    # 确定数值上下限
    L = np.exp(-2 / (n + 1))
    R = np.exp(2 / (n + 2))
    xmax = x0.max()
    xmin = x0.min()
    if xmin < 1:
        x0 += 1 - xmin
        c += 1 - xmin
    xmax = x0.max()
    xmin = x0.min()
    lambda_ = x0[0:-1] / x0[1:]  # 计算级比
    lambda_max = lambda_.max()
    lambda_min = lambda_.min()
    while lambda_max > R or lambda_min < L:
        x0 += xmin
        c += xmin
        xmax = x0.max()
        xmin = x0.min()
        lambda_ = x0[0:-1] / x0[1:]
        lambda_max = lambda_.max()
        lambda_min = lambda_.min()
    return c


# 灰色预测模型
def greyModel(dataVec, predictLen):
    "Grey Model for exponential prediction"
    # dataVec = [1, 2, 3, 4, 5, 6]
    # predictLen = 5

    x0 = np.array(dataVec, float)
    n = x0.shape[0]
    x1 = np.cumsum(x0)
    B = np.array([-0.5 * (x1[0:-1] + x1[1:]), np.ones(n - 1)]).T
    Y = x0[1:]
    u = linalg.lstsq(B, Y)[0]

    def diffEqu(y, t, a, b):
        return np.array(-a * y + b)

    t = np.arange(n + predictLen)
    sol = odeint(diffEqu, x0[0], t, args=(u[0], u[1]))
    sol = sol.squeeze()
    # print(sol)
    res = np.hstack((x0[0], np.diff(sol)))

    return res


def res_err(data, predata):
    len = data.size
    err = predata[:len] - data
    err = abs(err)
    # print(err)
    return err


def cal_times(data, predata):
    times_arr = np.zeros([2, 2], dtype=float)
    m = np.zeros(data.size)
    if data[0] - predata[0] >= 0:
        pre_state = 1
    else:
        pre_state = 0
    m[0] = 1 if pre_state == 1 else -1
    for i in range(data.size - 1):
        if data[i + 1] - predata[i + 1] >= 0:
            state = 1
        else:
            state = 0
        times_arr[1 - pre_state, 1 - state] += 1
        m[i + 1] = 1 if state == 1 else -1
        pre_state = state
    times_arr = times_arr.T / np.sum(times_arr, 1)
    init_state = np.array([pre_state, 1 - pre_state])
    return init_state, times_arr.T, m


def markov_err(init_state, state, step):
    # init_state = np.mat(init_state)
    # state = np.mat(state)
    m = np.zeros(step)
    init_state = init_state.T
    for i in range(step):
        result = np.dot(init_state, lng.matrix_power(state, i + 1))
        if result[0] > result[1]:
            m[i] = 1
        elif result[0] == result[1]:
            if i == 0:
                m[i] = -1 if init_state[0] == 0 else 1
            else:
                m[i] = m[i - 1]
        else:
            m[i] = -1

    return m


# 输入数据
x = [math.cos(i / 2) + 1 for i in range(300)]
x = np.array(x)
pred_len = 40
# print(x)
c = greyModelPreprocess(x)
x_hat = greyModel(x + c, pred_len) - c
x_pre = x_hat.copy()
err = res_err(x, x_hat)
x_var = np.var(err) / np.var(x)
c = greyModelPreprocess(err)
err_hat = greyModel(err + c, pred_len) - c
init_state, state, m = cal_times(x, x_hat)
mm = markov_err(init_state, state, pred_len)
# print(mm)

x_pre[: x.size] = x_pre[: x.size] + m * err_hat[: x.size]
x_pre[x.size :] = x_pre[x.size :] + mm * err_hat[x.size :]
# print(m)
# print(init_state)
# print(state)
pre_err = res_err(x, x_pre)
pre_var = np.var(pre_err) / np.var(x)
print(x_var)
print(pre_var)
# 画图
t1 = range(x.size)
t2 = range(x_hat.size)
plt.figure(1)
plt.plot(t1, x, color="r", linestyle="-", marker="*", label="True")
plt.plot(t2, x_hat, color="b", linestyle="--", marker=".", label="No Markov")
plt.plot(t2, x_pre, color="g", linestyle="--", marker="o", label="With Markpv")
plt.legend(loc="upper right")
plt.xlabel("xlabel")
plt.ylabel("ylabel")
plt.title("Prediction by Grey Model (GM(1,1))")
plt.show()
# plt.figure(2)
# t3 = range(err.size)
# t4 = range(err_hat.size)
# plt.plot(t3, err, color="r", linestyle="-", marker="*", label="True")
# plt.plot(t4, err_hat, color="b", linestyle="--", marker=".", label="Predict")
# plt.legend(loc="upper right")
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
# plt.title("Prediction by Grey Model (GM(1,1))")
# plt.show()
