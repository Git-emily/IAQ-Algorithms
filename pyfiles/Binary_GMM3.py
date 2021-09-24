import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


# reload(sys)
# sys.setdefaultencoding('utf-8')

class Binary_GMM():
    def __init__(self, iter_time, data_array, Mu_1, Sigma_1, Mu_2, Sigma_2, Mu_3, Sigma_3, Pi_weight, height, width):
        self.iter_time = iter_time
        self.Data = data_array
        self.mu_1 = Mu_1
        self.sigma_1 = Sigma_1
        self.mu_2 = Mu_2
        self.sigma_2 = Sigma_2
        self.mu_3 = Mu_3
        self.sigma_3 = Sigma_3
        self.pw = Pi_weight
        self.height = height
        self.width = width
        self.esp = 0.0001

    def PDF(self, data, Mu, sigma):
        """
        二元正态分布概率密度函数
        :param data: 一个二维数据点,ndarray
        :param Mu: 均值,ndarray
        :param Sigama: 协方差阵ndarray
        :return:该数据点的概率密度值
        """
        sigma_sqrt = math.sqrt(np.linalg.det(sigma))  # 协方差矩阵绝对值的1/2次
        sigma_inv = np.linalg.inv(sigma)  # 协方差矩阵的逆
        data.shape = (2, 1)
        Mu.shape = (2, 1)
        minus_mu = data - Mu
        minus_mu_trans = np.transpose(minus_mu)
        res = (1.0 / (2.0 * math.pi * sigma_sqrt)) * math.exp(
            (-0.5) * (np.dot(np.dot(minus_mu_trans, sigma_inv), minus_mu)))
        return res

    def E_step(self):
        """
        E-step: compute responsibilities
        计算出本轮gama_list
        :param Data:一系列二维的数据点
        :return:gama_list
        """
        gama_list = []
        for i, point in enumerate(self.Data):
            gama_i = []
            a = self.PDF(point, self.mu_1, self.sigma_1)  # 概率密度函数
            b = self.PDF(point, self.mu_2, self.sigma_2)
            c = self.PDF(point, self.mu_3, self.sigma_3)
            gama_i.append((self.pw[0] * a) / (self.pw[0] * a + self.pw[1] * b + self.pw[2] * c))
            gama_i.append((self.pw[1] * b) / (self.pw[0] * a + self.pw[1] * b + self.pw[2] * c))
            gama_i.append((self.pw[2] * c) / (self.pw[0] * a + self.pw[1] * b + self.pw[2] * c))
            gama_list.append(gama_i)
        return gama_list

    def M_step(self, gama_list):
        """
        M-step: compute weighted means and variances
        更新均值与协方差矩阵
        在此例中，   gama_i对应Mu_2,Var_2
                    (1-gama_i)对应Mu_1,Var_1
        :param X:一系列二维的数据点
        :return:
        """
        new_pi = []
        N_1 = 0.0
        N_2 = 0.0
        N_3 = 0.0
        for i, item in enumerate(gama_list):
            N_1 += item[0]
            N_2 += item[1]
            N_3 += item[2]

        # 更新均值
        new_mu_1 = np.array([0, 0])
        new_mu_2 = np.array([0, 0])
        new_mu_3 = np.array([0, 0])
        for i, item in enumerate(gama_list):
            new_mu_1 = new_mu_1 + self.Data[i] * item[0] / N_1
            new_mu_2 = new_mu_2 + self.Data[i] * item[1] / N_2
            new_mu_3 = new_mu_3 + self.Data[i] * item[2] / N_3

        # 很重要，numpy对一维向量无法转置，必须指定shape
        new_mu_1.shape = (2, 1)
        new_mu_2.shape = (2, 1)
        new_mu_3.shape = (2, 1)

        new_sigma_1 = np.array([[0, 0], [0, 0]])
        new_sigma_2 = np.array([[0, 0], [0, 0]])
        new_sigma_3 = np.array([[0, 0], [0, 0]])
        for i, item in enumerate(gama_list):
            data_tmp = [0, 0]
            data_tmp[0] = self.Data[i][0]
            data_tmp[1] = self.Data[i][1]
            vec_tmp = np.array(data_tmp)
            vec_tmp.shape = (2, 1)
            new_sigma_1 = new_sigma_1 + np.dot((vec_tmp - new_mu_1), (vec_tmp - new_mu_1).transpose()) * item[0] / N_1
            new_sigma_2 = new_sigma_2 + np.dot((vec_tmp - new_mu_2), (vec_tmp - new_mu_2).transpose()) * item[1] / N_2
            new_sigma_3 = new_sigma_2 + np.dot((vec_tmp - new_mu_2), (vec_tmp - new_mu_2).transpose()) * item[2] / N_3
            # print np.dot((vec_tmp-new_mu_1), (vec_tmp-new_mu_1).transpose())
        new_pi.append(N_1 / len(gama_list))
        new_pi.append(N_2 / len(gama_list))
        new_pi.append(N_3 / len(gama_list))

        # 更新类变量
        self.mu_1 = new_mu_1
        self.mu_2 = new_mu_2
        self.mu_3 = new_mu_3
        self.sigma_1 = new_sigma_1
        self.sigma_2 = new_sigma_2
        self.sigma_3 = new_sigma_3
        self.pw = new_pi

    # def EM_iterate(self):
    #     """
    #     EM算法迭代运行
    #     :param iter_time: 迭代次数，若为None则迭代至约束esp为止
    #     :param Data:数据
    #     :param esp:终止约束
    #     :return:
    #     """
    #     if self.iter_time == None:
    #         while (True):
    #             old_mu_1 = parameter_dict["Mu_1"].copy()
    #             old_mu_2 = parameter_dict["Mu_2"].copy()
    #             E_step(Data)
    #             M_step(Data)
    #             delta_1 = parameter_dict["Mu_1"] - old_mu_1
    #             delta_2 = parameter_dict["Mu_2"] - old_mu_2
    #             if math.fabs(delta_1[0]) < esp and math.fabs(delta_1[1]) < esp and math.fabs(
    #                     delta_2[0]) < esp and math.fabs(delta_2[1]) < esp:
    #                 break
    #     else:
    #         for i in range(iter_time):
    #             pass

    def EM_iterate_trajectories(self):
        """
        EM算法迭代运行,同时画出两个均值变化的轨迹
        :param iter_time:迭代次数，若为None则迭代至约束esp为止
        :param Data: 数据
        :param esp: 终止约束
        :return:
        """
        mean_trace_1 = [[], []]
        mean_trace_2 = [[], []]
        mean_trace_3 = [[], []]

        if self.iter_time == None:
            while (True):
                old_mu_1 = self.mu_1.copy()
                old_mu_2 = self.mu_2.copy()
                old_mu_3 = self.mu_3.copy()
                gama_list = self.E_step()
                self.M_step(gama_list)
                delta_1 = self.mu_1 - old_mu_1
                delta_2 = self.mu_2 - old_mu_2
                delta_3 = self.mu_3 - old_mu_3

                mean_trace_1[0].append(self.mu_1[0][0])
                mean_trace_1[1].append(self.mu_1[1][0])
                mean_trace_2[0].append(self.mu_2[0][0])
                mean_trace_2[1].append(self.mu_2[1][0])
                mean_trace_3[0].append(self.mu_3[0][0])
                mean_trace_3[1].append(self.mu_3[1][0])
                if math.fabs(delta_1[0][0]) < self.esp and math.fabs(delta_1[1][0]) < self.esp and math.fabs(
                        delta_2[0][0]) < self.esp and math.fabs(delta_2[1][0]) < math.fabs(
                    delta_3[0][0]) < self.esp and math.fabs(delta_3[1][0]) < self.esp:
                    break
        else:
            for i in range(self.iter_time):
                pass

        print('1_mu1:', mean_trace_1[0][-1])
        print('1_mu2:', mean_trace_1[1][-1])
        print('2_mu1:', mean_trace_2[0][-1])
        print('2_mu2:', mean_trace_2[1][-1])
        print('3_mu1:', mean_trace_3[0][-1])
        print('3_mu2:', mean_trace_3[1][-1])
        plt.subplot()
        # plt.xlim(xmax=1200, xmin=400)
        # plt.ylim(ymax=10, ymin=0)
        # plt.xlabel("CO2")
        # plt.ylabel("PM2.5")
        plt.plot(mean_trace_1[0], mean_trace_1[1], 'r-')
        plt.plot(mean_trace_1[0], mean_trace_1[1], 'b^')
        plt.show()

        plt.subplot()
        # plt.xlim(xmax=800, xmin=400)
        # plt.ylim(ymax=10, ymin=0)
        # plt.xlabel("CO2")
        # plt.ylabel("PM2.5")
        plt.plot(mean_trace_2[0], mean_trace_2[1], 'r-')
        plt.plot(mean_trace_2[0], mean_trace_2[1], 'b^')
        plt.show()

        plt.subplot()
        # plt.xlim(xmax=800, xmin=400)
        # plt.ylim(ymax=10, ymin=0)
        # plt.xlabel("CO2")
        # plt.ylabel("PM2.5")
        plt.plot(mean_trace_3[0], mean_trace_3[1], 'r-')
        plt.plot(mean_trace_3[0], mean_trace_3[1], 'b^')
        plt.show()

    # def EM_iterate_times(Data, mu_1, sigma_1, mu_2, sigma_2, pi_weight, esp=0.0001):
    #     # 返回迭代次数
    #     set_parameter(mu_1, sigma_1, mu_2, sigma_2, pi_weight)
    #     iter_times = 0
    #     while (True):
    #         iter_times += 1
    #         old_mu_1 = parameter_dict["Mu_1"].copy()
    #         old_mu_2 = parameter_dict["Mu_2"].copy()
    #         E_step(Data)
    #         M_step(Data)
    #         delta_1 = parameter_dict["Mu_1"] - old_mu_1
    #         delta_2 = parameter_dict["Mu_2"] - old_mu_2
    #         if math.fabs(delta_1[0]) < esp and math.fabs(delta_1[1]) < esp and math.fabs(
    #                 delta_2[0]) < esp and math.fabs(delta_2[1]) < esp:
    #             break
    #     return iter_times

    # def task_2():
    #     """
    #     执行50次，看迭代次数的分布情况
    #     这里协方差矩阵都取[[10, 0], [0, 10]]
    #     mean值在一定范围内随机生成50组数
    #     :return:
    #     """
    #     # 读取数据，猜初始值,执行算法
    #     Data_list = []
    #     with open("old_faithful_geyser_data.txt", 'r') as in_file:
    #         for line in in_file.readlines():
    #             point = []
    #             point.append(float(line.split()[1]))
    #             point.append(float(line.split()[2]))
    #             Data_list.append(point)
    #     Data = np.array(Data_list)
    #
    #     try:
    #         # 在10以内猜x1，在100以内随机取x2
    #         x_11 = 5
    #         x_12 = 54
    #         x_21 = 2
    #         x_22 = 74
    #         Mu_1 = np.array([x_11, x_12])
    #         Sigma_1 = np.array([[10, 0], [0, 10]])
    #         Mu_2 = np.array([x_21, x_22])
    #         Sigma_2 = np.array([[10, 0], [0, 10]])
    #         Pi_weight = 0.5
    #         iter_times = EM_iterate_times(Data, Mu_1, Sigma_1, Mu_2, Sigma_2, Pi_weight)
    #         print(iter_times)
    #     except Exception as e:
    #         print(e)


if __name__ == "__main__":
    data1 = []
    data2 = []
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    file_name1 = 'anomaly_data630094_Per15_Weekall_2021-02-28_2021-08-22 - week.csv'
    file_path1 = PATH + '\\Analysis_CO2\\' + file_name1
    with open(file_path1) as f:
        cols1 = len(f.readline().split(','))
    dataMat1 = np.genfromtxt(file_path1, delimiter=',', skip_header=0,
                             usecols=range(0, cols1 - 1))  # , skip_header=1, usecols=range(1, cols - 1)
    df_data = pd.DataFrame(dataMat1, columns=['Week', 'Time'])  # ,columns=['Time','Value']

    # file_name2 = '630094_Per15_Weekall_2020-12-29_2021-02-05.csv'
    # file_path2 = PATH + '\\Analysis_PM25\\' + file_name1
    # with open(file_path2) as f:
    #     cols2 = len(f.readline().split(','))
    # dataMat2 = np.genfromtxt(file_path2, delimiter=',', skip_header=1,usecols=range(1, cols1 - 1))
    # for i in range(dataMat2.shape[0] - 1):
    #     data2 = np.concatenate((data2, dataMat2[i].tolist()), axis=0)
    #
    # data_array = np.array([data1.ravel(), data2.ravel()]).T
    #
    # df_data  = pd.DataFrame(data_array,columns=['CO2','PM2.5'])

    # plot rawdata
    sn.jointplot(x='Time', y='Week', data=df_data)
    plt.show()
    plt.savefig(PATH + '\\No_Kind_Joint.png')

    sn.jointplot(x='Time', y='Week', data=df_data, kind='kde')
    plt.show()
    plt.savefig(PATH + '\\KDE_Joint.png')

    # fig = plt.figure()
    # ax= Axes3D(fig)
    # ax.plot_surface()

    Mu_1 = np.array([6, 72])
    Sigma_1 = np.array([[10, 0], [0, 10]])
    Mu_2 = np.array([4, 46])
    Sigma_2 = np.array([[10, 0], [0, 10]])
    Mu_3 = np.array([2, 17])
    Sigma_3 = np.array([[10, 0], [0, 10]])
    Pi_weight = [1 / 3, 1 / 3, 1 / 3]
    Bin_GMM = Binary_GMM(iter_time=None, data_array=dataMat1, Mu_1=Mu_1, Sigma_1=Sigma_1, Mu_2=Mu_2, Sigma_2=Sigma_2,
                         Mu_3=Mu_3, Sigma_3=Sigma_3, Pi_weight=Pi_weight, height=dataMat1.shape[0],
                         width=dataMat1.shape[1])
    Bin_GMM.EM_iterate_trajectories()
    # task_2()
