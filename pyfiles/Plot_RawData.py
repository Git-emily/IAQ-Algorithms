import os

import matplotlib.pyplot as plt
import numpy as np


class TGMM():
    def __init__(self, group, height, all_height, all_width, data_array, PATH):
        self.group = group
        self.height = height
        self.all_height = all_height
        self.all_width = all_width
        self.data_array = data_array
        self.raw_data = np.zeros((self.height, self.all_width))
        self.data_list = []
        self.PATH = PATH

    def data_pro(self):
        i = 0
        while (i < self.all_height):
            for j in range(self.height):
                for k in range(self.all_width):
                    z = self.data_array[j, k]
                    self.data_list.append([j, k, z])
            i += 1
        print(type(self.data_list))

    def data_plot(self):
        # for list in self.data_list:
        X = [list[0] for list in self.data_list]
        Y = [list[1] for list in self.data_list]
        Z = [list[2] for list in self.data_list]
        # fig = plt.figure()  # 创建一个三维的绘图工程
        ax = plt.subplot(projection='3d')
        ax.set_title('Anomaly_Count_Data')  # 设置本图名称
        ax.scatter(X, Y, Z, c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
        plt.show()
        if not os.path.exists(self.PATH + r'\export_Emily'):
            os.mkdir(self.PATH + r'\export_Emily')
        plt.savefig(os.path.join(self.PATH + r'\export_Emily', "RawData_Plot.png"),
                    format='png')  # dpi=600, bbox_inches='tight'


def main(group, height, all_height, all_width, data_array, PATH):
    T_GMM = TGMM(group=group, height=height, all_height=all_height, all_width=all_width, data_array=data_array,
                 PATH=PATH)
    T_GMM.data_pro()
    T_GMM.data_plot()
    # Plot_Data(data_array)


if __name__ == '__main__':
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    file_name = '630094_Per15_Weekall_2021-02-28_2021-08-22 - Copy (2).csv'
    file_path = PATH + '\\Analysis_CO2\\' + file_name
    with open(file_path) as f:
        cols = len(f.readline().split(','))
    data_array = np.genfromtxt(file_path, delimiter=',', skip_header=1,
                               usecols=range(2, cols - 2))  # , skip_header = 1, usecols=range(1,cols-1)
    all_height = data_array.shape[0]
    all_width = data_array.shape[1]
    height = 7
    group = all_height // height + 1
    main(group, height, all_height, all_width, data_array, PATH)
