# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
# import Plot_RawData
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MOG():
    def __init__(self, numOfGauss, meanVal, varVal, BG_thresh, ncols, lr, ashrae_standare, variable, file_path,
                 file_name, PATH):
        self.numOfGauss = numOfGauss
        self.BG_thresh = BG_thresh
        self.lr = lr
        self.ashrae_standare = ashrae_standare
        self.file_path = file_path
        self.file_name = file_name
        self.PATH = PATH
        self.ncols = ncols
        self.variable = variable
        # self.data_df = pd.read_csv(self.file_path) #,usecols=range(2,self.ncols)
        # self.data_df['date'] = self.data_df['date'].apply(lambda x: int(time.mktime(time.strptime(x,'%Y/%m/%d'))))
        self.data_array = np.genfromtxt(self.file_path, delimiter=',', skip_header=1,
                                        usecols=range(0, self.ncols))  # , skip_header=1, usecols=range(2, self.ncols)
        # self.data_array = np.array(self.data_df)
        self.rows = self.data_array.shape[0]
        self.width = self.data_array.shape[1] - 1
        self.anomaly_list = []
        self.anomaly_count = []
        # self.mus=np.zeros((self.height,self.width, self.numOfGauss, 3)) ## assuming using color frames
        self.mus = np.zeros((self.width, self.numOfGauss))  ## assuming using gray-scale frames
        self.sigmaSQs = np.zeros((self.width,
                                  self.numOfGauss))  ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.width, self.numOfGauss))
        self.currentBG = np.zeros(self.width)
        # for i in range(self.height):
        for j in range(self.width):
            self.mus[j] = [meanVal] * self.numOfGauss
            self.sigmaSQs[j] = [varVal] * self.numOfGauss
            self.omegas[j] = [1.0 / self.numOfGauss] * self.numOfGauss

    def reorder(self):
        BG_pivot = np.zeros((self.width), dtype=int)
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
        for j in range(self.width):
            X = curFrame[j, 0, 2]
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
                rho = self.lr * 0.75  # use 0.75 to reduce computation time
                self.sigmaSQs[j, match] = (1.0 - rho) * self.sigmaSQs[j, match] + rho * np.dot(
                    (X - self.mus[j, match]).T, (X - self.mus[j, match]))
                self.mus[j, match] = (1.0 - rho) * self.mus[j, match] + rho * X
                ##label the pixel
                self.currentBG[j] = np.dot(self.omegas[j], self.mus[j, :])
                if X >= self.currentBG[j] and (match > BG_pivot[j]):  # or X >= self.ashrae_standare
                    labels[j] = 250
                    self.anomaly_list.append(list(curFrame[j]))
            else:
                # none match the current value, the least prob. distribution
                # is replaced with one with current value as its mean, an
                # initial high variance, and low prior weight
                self.mus[j, -1] = X
                self.sigmaSQs[j, -1] = 40000  # varVal or a high value
                self.omegas[j, -1] = 0.02  # a low prior weight
                labels[j] = 250
                self.anomaly_list.append(list(curFrame[j]))

            # re-normalizng the weights --- by Ziyou
            sum_omegas = sum(self.omegas[j])
            if (sum_omegas != 1.0):
                for k in range(self.numOfGauss):
                    self.omegas[j, k] = self.omegas[j, k] / sum_omegas
        # update by emily
        # for k, v in enumerate(labels):
        #     if k > 0:
        #         if labels[k] == 250:
        #             m += 1
        #             # if labels[k - 1] == labels[k]:
        #             if m >= 4 and labels[k + 1] != 250:
        #                 for i in range(m, 0, -1):
        #                     self.anomaly_list.append(curFrame[k - i + 1])
        #                 self.anomaly_count.append([curFrame[k,0,0],[curFrame[k-m+1,0,1],curFrame[k,0,1]]])
        #                 m = 0
        #             elif labels[k + 1] == 250:
        #                 continue
        #             elif m < 4 and labels[k + 1] != 250:
        #                 m = 0
        return labels

    def streamer(self):
        temp_data_array = np.zeros((self.rows, self.width, 1, 3))
        TotalNumFrames = np.size(self.data_array, 0)
        for i in range(TotalNumFrames):
            for j in range(self.width):
                x = self.data_array[i, 0]
                y = j
                z = self.data_array[i, j + 1]
                temp_array = np.array([x, y, z])
                temp_data_array[i, j] = temp_array
        TotalNumFrames = np.size(self.data_array, 0)
        for fr in range(TotalNumFrames):
            frame = temp_data_array[fr, :, :, :]
            print("number of frames: ", fr)
            print(frame[:, 0, 2].T)
            BG_pivots = self.reorder()
            labels = self.updateParam(frame, BG_pivots)
            # print(labels)
            # print(np.sum(labels[0, :] > 0.0))
            plt.figure(1)
            plt.plot(frame[:, 0, 2].T)
            plt.plot(labels[:], 'r')  # red color
            plt.axis([0, self.width, 400, 1000])

            # extract plot current BG
            plt.figure(1)
            plt.plot(self.currentBG[:], 'g')
            plt.axhline(y=600, color='orange', linestyle='-')
            plt.axis([0, self.width, 400, 1000])
            plt.show()
        print('Done')

    def anomaly_analysis(self):
        print('Start to analysis')
        anomaly_narray = np.array(self.anomaly_list)
        anomaly_array = np.reshape(anomaly_narray, (-1, 3))

        temp_anomaly_array = anomaly_array.T
        X = list(temp_anomaly_array[0])
        Y = list(temp_anomaly_array[1])
        Z = list(temp_anomaly_array[2])
        # ax = plt.subplot(111, projection='3d')
        #
        # ax.set_title('Plot_Anomaly_Data')  # 设置本图名称
        # ax.scatter(X, Y, Z, c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
        # plt.show()
        # print('Plot Done')

        analomy_DF = pd.DataFrame(anomaly_array, columns=['Timestamp', 'Time', 'Value'])
        analomy_DF['Date'] = analomy_DF['Timestamp'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
        analomy_DF = analomy_DF[['Timestamp', 'Time', 'Value']]
        anamaly_file = 'anomaly_data' + self.file_name
        anamaly_path = os.path.join(self.PATH, 'Analysis_' + self.variable, anamaly_file)
        np.savetxt(anamaly_path, analomy_DF, delimiter=',', fmt='%d')

        analomy_DF_week = pd.DataFrame(anomaly_array, columns=['Week', 'Time', 'Value'])
        analomy_DF_week = analomy_DF_week[['Week', 'Time', 'Value']]
        df_anomaly = analomy_DF_week.set_index('Week', drop=False)
        df_anomaly = df_anomaly.groupby(df_anomaly.index).agg({'Time': 'value_counts'})  # 'Week': 'first',
        df_anomaly = df_anomaly.unstack()
        anamaly_file = 'anomaly_Count' + self.file_name
        anamaly_path = os.path.join(self.PATH, 'Analysis_' + self.variable, anamaly_file)
        df_anomaly.to_csv(anamaly_path)

        data_array = df_anomaly.values
        # Plot_RawData.main(group=0, height=7, all_height=data_array.shape[0], all_width=data_array.shape[1], data_array=data_array,PATH=self.PATH)

        # analomy_DF = analomy_DF[['Timestamp','Time']]
        # analomy_DF = analomy_DF.set_index('Timestamp')
        # analomy_count_DF = analomy_DF.groupby('Timestamp').unstack()

        analomy = analomy_DF.pivot(index='Timestamp', columns='Time', values='Value')
        print(analomy)
        anamaly_file = 'anomaly_value' + self.file_name
        anamaly_path = os.path.join(self.PATH, 'Analysis_' + self.variable, anamaly_file)
        analomy.to_csv(anamaly_path)
        print('DONE')


def main():
    ## initialize pixel gaussians
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    file_name = '630094_Per15_Weekall_2021-02-28_2021-08-22 - week.csv'
    print(PATH)
    variable = 'CO2'
    file_path = PATH + r'\Analysis_' + variable + '\\' + file_name
    with open(file_path) as f:
        ncols = len(f.readline().split(','))
    subtractor = MOG(numOfGauss=4, meanVal=600, varVal=40000, BG_thresh=0.6, lr=0.1, ncols=ncols, ashrae_standare=0,
                     variable=variable, file_path=file_path, file_name=file_name, PATH=PATH)
    subtractor.streamer()
    subtractor.anomaly_analysis()


if __name__ == '__main__':
    main()
