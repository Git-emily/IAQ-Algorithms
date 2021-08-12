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
    def __init__(self, numOfGauss, meanVal, varVal, BG_thresh, lr, width,ashrae_standare):
        self.numOfGauss = numOfGauss
        self.BG_thresh = BG_thresh
        self.lr = lr
        # self.height=height
        self.width = width
        self.ashrae_standare = ashrae_standare
        # self.mus=np.zeros((self.height,self.width, self.numOfGauss, 3)) ## assuming using color frames
        self.mus = np.zeros((self.width, self.numOfGauss))  ## assuming using gray-scale frames
        # self.mus=np.zeros((self.height,self.width, self.numOfGauss, 1)) ## assuming using gray-scale frames
        self.sigmaSQs = np.zeros((self.width,
                                  self.numOfGauss))  ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.width, self.numOfGauss))
        self.currentBG = np.zeros(self.width)
        # for i in range(self.height):
        for j in range(self.width):
            # self.mus[i,j]=np.array([[meanVal, meanVal, meanVal]]*self.numOfGauss) ##assuming a 400ppm for CO2 mean
            self.mus[j] = [meanVal] * self.numOfGauss
            # self.mus[i,j]=np.array([[meanVal]]*self.numOfGauss) ##assuming a 400ppm for CO2 mean
            self.sigmaSQs[j] = [varVal] * self.numOfGauss
            self.omegas[j] = [1.0 / self.numOfGauss] * self.numOfGauss

    def reorder(self):
        BG_pivot = np.zeros((self.width), dtype=int)
        # for i in range(self.height):
        for j in range(self.width):
            BG_pivot[j] = -1
            ratios = []
            for k in range(self.numOfGauss):
                ratios.append(self.omegas[j, k] / np.sqrt(self.sigmaSQs[j, k]))
            indices = np.array(np.argsort(ratios)[::-1])
            self.mus[j] = self.mus[j][indices]
            self.sigmaSQs[j] = self.sigmaSQs[j][indices]
            self.omegas[j] = self.omegas[j][indices]
            cummProb = 0
            for l in range(self.numOfGauss):
                cummProb += self.omegas[j, l]
                if cummProb >= self.BG_thresh and l < self.numOfGauss - 1:
                    BG_pivot[j] = l
                    break
            ##if no background pivot is made the last one is foreground
            if BG_pivot[j] == -1:
                BG_pivot[j] = self.numOfGauss - 2
        return BG_pivot

    def updateParam(self, curFrame, BG_pivot):
        labels = np.zeros((self.width))
        # for i in range(self.height):
        for j in range(self.width):
            X = curFrame[0, j]
            match = -1                
            for k in range(self.numOfGauss):
                CoVarInv = np.linalg.inv(self.sigmaSQs[j, k] * np.eye(1))  # 计算矩阵的逆矩阵
                X_mu = X - self.mus[j, k]
                dist = np.dot(X_mu.T, np.dot(CoVarInv, X_mu))
                # if dist<6.25*self.sigmaSQs[i,j,k]: #6.25 = 2.5^2; 9.0=3.0^2
                if dist < 6.25:
                    match = k
                    break
            
            if match != -1:  ## a match found
                ##update parameters
                self.omegas[j] = (1.0 - self.lr) * self.omegas[j]
                self.omegas[j, match] += self.lr
                sum_w = sum(self.omegas[j]) 
                rho = self.lr/sum_w
                
                #rho=self.lr * multivariate_normal.pdf(X,self.mus[j,match],np.linalg.inv(CoVarInv))
                #rho = self.lr * 0.75;  # use 0.75 to reduce computation time
                self.sigmaSQs[j, match] = (1.0 - rho) * self.sigmaSQs[j, match] + rho * np.dot(
                    (X - self.mus[j, match]).T, (X - self.mus[j, match]))
                self.mus[j, match] = (1.0 - rho) * self.mus[j, match] + rho * X
                ##label the pixel
                if match > BG_pivot[j] or X >= self.ashrae_standare:
                    labels[j] = 250
            else:
                # none match the current value, the least prob. distribution
                # is replaced with one with current value as its mean, an
                # initial high variance, and low prior weight
                self.mus[j, -1] = X
                self.sigmaSQs[j, -1] = 40000  # varVal or a high value
                self.omegas[j, -1] = 0.02  # a low prior weight
                labels[j] = 250
            # re-normalizng the weights --- by Ziyou
            sum_omegas = sum(self.omegas[j])
            if (sum_omegas != 1.0):
                for k in range(self.numOfGauss):
                    self.omegas[j, k] = self.omegas[j, k] / sum_omegas

        return labels

    def extractCurrentBG(self):
        # for i in range(self.height):
        for j in range(self.width):
            # self.currentBG[i, j] = np.dot(self.omegas[i, j], self.mus[i, j, :, 0])
            self.currentBG[j] = np.dot(self.omegas[j], self.mus[j, :])

    def streamer(self):
        ## initialize pixel gaussians
        # the *.csv files are generated from D:\xiongz\2020\A2L_Projects\FieldTrialOtherUnits\dataFigureOf26units\readDataFromFigure_plot_daily_CO2_data.m
        # data_array = np.genfromtxt('630091_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        # data_array = np.genfromtxt('630094_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        # data_array = np.genfromtxt('62ff20_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        # data_array = np.genfromtxt('6300c7_dailyData_Dec292020_Feb062021.csv', delimiter=',')
        PATH = os.path.abspath(os.path.dirname(os.getcwd()))
        file_name = '630094_dailyData_Dec292020_Feb062021.csv'
        # file_name = "61a3fa_dailyData_Dec292020_Feb062021.csv"
        print(PATH)
        file_path = PATH + r'\Raw_data' + '\\' + file_name
        data_array = np.genfromtxt(file_path, delimiter=',')
        TotalNumFrames = np.size(data_array, 0)
        data_array = np.expand_dims(data_array, axis=1)
        for fr in range(TotalNumFrames):
            frame = data_array[fr, :, :]

            print("number of frames: ", fr)
            print(frame[0, :])
            print(type(frame[0, :]))
            BG_pivots = self.reorder()
            labels = self.updateParam(frame, BG_pivots)
            # print(labels)
            # print(np.sum(labels[0, :] > 0.0))
            plt.figure(1)
            plt.plot(frame[0, :])
            plt.plot(labels[:], 'r')  # red color
            plt.axis([0, 17500, 0, 1600])

            # extract plot current BG
            self.extractCurrentBG()
            plt.figure(1)
            plt.plot(self.currentBG[:], 'g')
            # plt.axis([0, 17500, 400, 1600])
            plt.show()


def main():
    subtractor = MOG(numOfGauss=4, meanVal=600.0, varVal=40000.0, BG_thresh=0.6, lr=0.026, width=16384, ashrae_standare = 1000)  # note to change width accordingly
    subtractor.streamer()


if __name__ == '__main__':
    main()
