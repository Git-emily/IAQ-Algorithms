# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class MOG():
    def __init__(self, numOfGauss, meanVal, varVal, BG_thresh, lr, height, width, file_path):
        self.file_path = file_path
        self.numOfGauss = numOfGauss
        self.BG_thresh = BG_thresh
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height, self.width, self.numOfGauss, 3))  ## assuming using color frames
        # self.mus=np.zeros((self.height,self.width, self.numOfGauss)) ## assuming using gray-scale frames
        self.sigmaSQs = np.zeros((self.height, self.width,
                                  self.numOfGauss))  ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.height, self.width, self.numOfGauss))
        self.currentBG = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i, j] = np.array(meanVal * self.numOfGauss)  # 混合高斯的初始均值
                # self.mus[i,j]=np.array(meanVal) ##assuming a 400ppm for CO2 mean
                self.sigmaSQs[i, j] = [varVal] * self.numOfGauss  # 混合高斯的初始标准差
                self.omegas[i, j] = [1.0 / self.numOfGauss] * self.numOfGauss  # 各高斯模型的系数

    def reorder(self):
        BG_pivot = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                BG_pivot[i, j] = -1
                ratios = []
                for k in range(self.numOfGauss):
                    ratios.append(self.omegas[i, j, k] / np.sqrt(self.sigmaSQs[i, j, k]))
                indices = np.array(np.argsort(ratios)[::-1])
                self.mus[i, j] = self.mus[i, j][indices]
                self.sigmaSQs[i, j] = self.sigmaSQs[i, j][indices]
                self.omegas[i, j] = self.omegas[i, j][indices]
                cummProb = 0
                for l in range(self.numOfGauss):
                    cummProb += self.omegas[i, j, l]
                    if cummProb >= self.BG_thresh and l < self.numOfGauss - 1:
                        BG_pivot[i, j] = l
                        break
                ##if no background pivot is made the last one is foreground
                if BG_pivot[i, j] == -1:
                    BG_pivot[i, j] = self.numOfGauss - 2
        return BG_pivot

    def updateParam(self, curFrame, BG_pivot):
        labels = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                X = curFrame[i, j]
                match = -1
                for k in range(self.numOfGauss):
                    CoVarInv = np.linalg.inv(self.sigmaSQs[i, j, k] * np.eye(3))
                    X_mu = X - self.mus[i, j, k]
                    dist = np.dot(X_mu.T, np.dot(CoVarInv, X_mu))
                    if dist < 6.25 * self.sigmaSQs[i, j, k]:  # 6.25 = 2.5^2; 9.0=3.0^2
                        match = k
                        break
                if match != -1:  ## a match found
                    ##update parameters
                    self.omegas[i, j] = (1.0 - self.lr) * self.omegas[i, j]
                    self.omegas[i, j, match] += self.lr
                    rho = self.lr * multivariate_normal.pdf(X, self.mus[i, j, match], np.linalg.inv(CoVarInv))
                    self.sigmaSQs[i, j, match] = (1.0 - rho) * self.sigmaSQs[i, j, match] + rho * np.dot(
                        (X - self.mus[i, j, match]).T, (X - self.mus[i, j, match]))
                    self.mus[i, j, match] = (1.0 - rho) * self.mus[i, j, match] + rho * X
                    ##label the pixel
                    if match > BG_pivot[i, j]:
                        labels[i, j] = 250
                else:
                    self.mus[i, j, -1] = X
                    labels[i, j] = 250
        return labels

    def extractCurrentBG(self):
        for i in range(self.height):
            for j in range(self.width):
                self.currentBG[i, j] = np.dot(self.omegas[i, j], self.mus[i, j, :, 0])

    def streamer(self):
        ## initialize pixel gaussians
        try:
            data_array = np.genfromtxt(self.file_path, delimiter=',', skip_header=1, usecols=range(2, self.width + 2))
        except Exception as e:
            print('E:', e)
        # data_array = np.genfromtxt('630094_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        try:
            TotalNumFrames = np.size(data_array, 0)
        except Exception as e:
            print('E2', e)
        data_array = np.expand_dims(data_array, axis=1)
        for fr in range(TotalNumFrames):
            frame = data_array[fr, :, :]
            test = frame[0:]
            print("number of frames: ", fr)
            print(frame[0, :])
            BG_pivots = self.reorder()
            labels = self.updateParam(frame, BG_pivots)
            # print(labels)
            # print(np.sum(labels[0, :] > 0.0))
            plt.figure(1)
            plt.plot(frame[0, :])
            plt.plot(labels[0, :], 'r')  # red color
            plt.axis([0, self.width, 0, 1200])

            plt.show()

            # extract plot current BG
            self.extractCurrentBG()
            plt.figure(1)
            plt.plot(self.currentBG[0, :], 'g')
            plt.axis([0, self.width, 0, 1200])
            plt.show()


def main():
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    file_name = '630094_Per15_Monday_2021-02-28_2021-08-22.csv'
    print(PATH)
    file_path = PATH + r'\Analysis_CO2' + '\\' + file_name
    subtractor = MOG(numOfGauss=4, meanVal=600.0, varVal=40000.0, BG_thresh=0.6, lr=0.1, height=1, width=96,
                     file_path=file_path)  # note to change width accordingly
    subtractor.streamer()


if __name__ == '__main__':
    main()
