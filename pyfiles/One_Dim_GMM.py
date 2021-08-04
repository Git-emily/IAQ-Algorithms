# -*- coding: utf-8 -*-
# for multi-Gaussian
__author__ = "Emily"

from numpy.linalg import det, inv
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
import os
import numpy as np


class MOG():
    def __init__(self, numOfGauss, meanVal, bias, varVal, file_path):
        self.file_path = file_path
        self.numOfGauss = numOfGauss
        self.bias = bias
        # 设置均值
        self.phais = [1.0 / self.numOfGauss] * self.numOfGauss # 各高斯模型的系数
        self.mus = np.array([meanVal]* self.numOfGauss) * np.array(self.bias) # 混合高斯的初始均值
        self.sigmas = [varVal] * self.numOfGauss  # 混合高斯的初始标准差
        self.data_array = np.genfromtxt(file_path, delimiter=',')
        self.data = []
        for i in range(self.data_array.shape[0]-1):
            self.data = np.concatenate((self.data_array[i], self.data_array[i+1]), axis=0)
        self.data = self.data[np.logical_not(np.isnan(self.data))]

    def gaussian(self,x,mu,sigma):
        temp = -np.square(x-mu)/(2*sigma)
        return np.exp(temp)/(np.sqrt(2.0*np.pi*sigma)) # sigma = sigma^2

    def e_step(self):
        Qs = []
        for i in range(len(self.data)):
            q = [phai*self.gaussian(self.data[i], mu, sigma) for phai,mu,sigma in zip(self.phais,self.mus,self.sigmas)]
            Qs.append(q)
        Qs = np.array(Qs)
        Qs = Qs / np.sum(Qs,axis=1).reshape(-1,1)  # axis=1 working along the row
        return Qs

    def m_step(self):
        data = np.array(self.data)
        gama_j = np.sum(self.Qs,axis=0)
        new_phais = gama_j/len(self.data)
        # print("new_phai:",new_phais)
        mu_temp = np.sum(self.Qs*(data.reshape(-1,1)),axis=0)
        new_mus =mu_temp/gama_j
        X_i_mu_j = np.square(np.array([data]).reshape(-1,1)-np.array([self.mus]))
        new_sigmas = np.sum(self.Qs*X_i_mu_j,axis=0)/gama_j
        return new_phais,new_mus,new_sigmas

    def EM(self):
        #plot initial data
        # sn.distplot(self.data,kde=True,axlabel='CO2')

        # 开始学习
        for i in range(200):  # 需要考虑迭代结束后值是收敛的
            self.Qs = self.e_step()
            self.phais, self.mus, self.sigmas= self.m_step()
            print('New_Para',self.phais,self.mus,self.sigmas)
        print('Final_Para',self.phais,self.mus,self.sigmas)

        print('Start to plot')
        # plt.figure(1)
        # plt.plot(self.data)
        samples = [subtractor.gaussian_mixture(x) for x in self.data]
        SUM = sum(samples)
        print(SUM)
        # for i in range(len(self.data)):
        #     Z = np.random.choice([-1,0,1])
        #     samples.append(np.random.normal(self.mus[Z],self.sigmas[Z],1))
        plt.plot(self.data.tolist(),samples)
        plt.show()
        # if not os.path.exists(PATH + r'\export_Emily'):
        #     os.mkdir(PATH + r'\export_Emily')
        # plt.savefig(os.path.join(PATH + r'\export_Emily',  "One_Dim_GMM.png"), format='png')  # dpi=600, bbox_inches='tight'
        # sn.distplot(samples)
        # plt.show()

    def gaussian_mixture(self,x):
        z = 0
        for idx in range(len(self.phais)):
            # dim = len(x)
            constant = (2 * np.pi) ** (-1 / 2) * (self.sigmas[idx]) ** (-0.5)
            p = constant * np.exp(-0.5 * ((x - self.mus[idx])**2) *(2 * self.sigmas[idx])**-1)

            z += self.phais[idx] * p
            # subtractor.gaussion(x,self.mus[idx],self.sigmas[idx])
        return z

    def gaussion(x, mu,sigma):
        dim = len(x)
        constant = (2*np.pi) ** (-dim/2) * det(sigma) **(-0.5)
        return constant * np.exp(-0,5 * (x-mu).dot(inv(sigma)).dot(x-mu))


if __name__ == '__main__':
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    # file_name = '630091_dailyData_Dec292020_Feb062021.csv'
    file_name = "61a3fa_dailyData_Dec292020_Feb062021.csv"
    print(PATH)
    file_path = PATH + r'\Raw_data' + '\\' + file_name
    subtractor = MOG(numOfGauss=3, meanVal=400.0,  bias = [1.3, 2, 2.5], varVal=200.0, file_path=file_path)
    subtractor.EM()
    # subtractor.streamer()