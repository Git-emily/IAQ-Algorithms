import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det


# 多维高斯分布
def gaussion(x, mu, Sigma):
    dim = len(x)
    constant = (2 * np.pi) ** (-dim / 2) * det(Sigma) ** (-0.5)
    return constant * np.exp(-0.5 * (x - mu).dot(inv(Sigma)).dot(x - mu))


# 混合高斯模型
def gaussion_mixture(x, Pi, mu, Sigma):
    z = 0
    for idx in range(len(Pi)):
        z += Pi[idx] * gaussion(x, mu[idx], Sigma[idx])
    print('Z:', z)
    return z


#
# Pi = [ 0.4, 0.6 ]
# mu = [ np.array([1,1]), np.array([-1,-1]) ]
# Sigma = [ np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]) ]
#
# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
# x, y = np.meshgrid(x, y)
#
# X = np.array([x.ravel(), y.ravel()]).T
# z = [gaussion_mixture(x, Pi, mu, Sigma) for x in X]
# z = np.array(z).reshape(x.shape)
#
# fig = plt.figure()
# # 绘制3d图形
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1.plot_surface(x, y, z)
# # 绘制等高线
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.contour(x, y, z)
#
# plt.show()

if __name__ == "__main__":
    data1 = []
    data2 = []
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    file_name1 = '630094_Per15_Weekall_2020-12-29_2021-02-05.csv'
    file_path1 = PATH + '\\Analysis_CO2\\' + file_name1
    with open(file_path1) as f:
        cols1 = len(f.readline().split(','))
    dataMat1 = np.genfromtxt(file_path1, delimiter=',', skip_header=1,
                             usecols=range(1, cols1 - 1))  # , skip_header=1, usecols=range(1, cols - 1)
    for i in range(dataMat1.shape[0] - 1):
        data1 = np.concatenate((data1, dataMat1[i].tolist()), axis=0)
    x = np.array(data1)

    file_name2 = '630094_Per15_Weekall_2020-12-29_2021-02-05.csv'
    file_path2 = PATH + '\\Analysis_PM25\\' + file_name1
    with open(file_path2) as f:
        cols2 = len(f.readline().split(','))
    dataMat2 = np.genfromtxt(file_path2, delimiter=',', skip_header=1, usecols=range(1, cols1 - 1))
    for i in range(dataMat2.shape[0] - 1):
        data2 = np.concatenate((data2, dataMat2[i].tolist()), axis=0)
    y = np.array(data2)

    x, y = np.meshgrid(x, y)
    Data = np.array([x.ravel(), y.ravel()]).T

    Pi = [0.5, 0.5]
    mu = [np.array([500, 1]), np.array([800, 2])]
    Sigma = [np.array([[500, 0], [0, 500]]), np.array([[500, 0], [0, 500]])]

    z = [gaussion_mixture(x, Pi, mu, Sigma) for x in Data]
    z = np.array(z).reshape(x.shape)

    fig = plt.figure()
    # 绘制3d图形
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(x, y, z)
    # 绘制等高线
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(x, y, z)

    plt.show()
