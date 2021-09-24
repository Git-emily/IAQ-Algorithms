import os

import numpy as np


def gaussian(x, mu, sigma):
    temp = -np.square(x - mu) / (2 * sigma)
    return np.exp(temp) / (np.sqrt(2.0 * np.pi * sigma))  # sigma = sigma^2


def e_step(data, phais, mus, sigmas):
    Qs = []
    for i in range(len(data)):
        q = [phai * gaussian(data[i], mu, sigma) for phai, mu, sigma in zip(phais, mus, sigmas)]
        Qs.append(q)
    Qs = np.array(Qs)
    Qs = Qs / np.sum(Qs, axis=1).reshape(-1, 1)
    return Qs


def m_step(data, phais, mus, sigmas, Qs):
    data = np.array(data)
    gama_j = np.sum(Qs, axis=0)
    new_phais = gama_j / len(data)
    print("new_phai:", new_phais)
    mu_temp = np.sum(Qs * (data.reshape(-1, 1)), axis=0)
    new_mus = mu_temp / gama_j
    X_i_mu_j = np.square(np.array([data]).reshape(-1, 1) - np.array([mus]))
    new_sigmas = np.sum(Qs * X_i_mu_j, axis=0) / gama_j
    return new_phais, new_mus, new_sigmas


def EM(data, k=1):
    # 设置均值
    phais = [1.0 / k for i in range(k)]  # 各高斯模型的系数
    mus = [400 for i in range(k)]  # 混合高斯的初始均值
    sigmas = [2000 for i in range(k)]  # 混合高斯的初始标准差
    # 开始学习
    for i in range(100):
        Qs = e_step(data, phais, mus, sigmas)
        phais, mus, sigmas = m_step(data, phais, mus, sigmas, Qs)
        print(phais, mus, sigmas)


if __name__ == "__main__":
    s1 = np.random.normal(19, 1, 10000)
    s2 = np.random.normal(1, 1, 10000)
    s3 = np.concatenate((s1, s2))
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    # file_name = '630091_dailyData_Dec292020_Feb062021.csv'
    file_name = "630094_dailyData_Dec292020_Feb062021.csv"
    print(type(s1))
    file_path = PATH + r'\Raw_data' + '\\' + file_name
    data_array = np.genfromtxt(file_path, delimiter=',')
    for i in range(0, data_array.shape[0] - 1, 2):
        data = np.concatenate((data_array[i], data_array[i + 1]), axis=0)
    print(type(data))
    EM(data, 2)
