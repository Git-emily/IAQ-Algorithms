import os
import time

import numpy as np
import pandas as pd

Evaluation = pd.DataFrame(columns=['Date', 'StartTime', 'EndTime'])
data_list = []
StartTime = []
EndTime = []


def read_txt(file_path):
    i = 0
    i_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            # print(line)
            if not 'PM1' in line:
                Starttime = line.split(' ')[0][:-3]
                Endtime = line.split(' ')[1][:-3]
                StartTime.append(Starttime)
                EndTime.append(Endtime)
                i += 1
            else:
                i_list.append(i)
                if len(i_list) > 1:
                    for k in range(i_list[-1] - i_list[-2] - 1):
                        data_list.append(data_list[-1])
                front = line.split('.')[0]
                Date = front[-10:].replace('_', '-')
                data_list.append(Date)

        data_list.append(data_list[-1])
        data_list.append(data_list[-1])
        data_list.append(data_list[-1])

        Evaluation['Date'] = data_list
        Evaluation['StartTime'] = StartTime
        Evaluation['EndTime'] = EndTime
        Evaluation['Value_Count'] = 1
    return Evaluation


def Evaluation_Process(PATH, Evaluation, analomy_data):
    Evaluation['Timestamp'] = Evaluation['Date'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d'))))

    # Evaluation['Start'] = Evaluation['StartTime'].apply(lambda x: int((float(x.split(':')[0])*60)+float(x.split(':')[1])))
    # Evaluation['End'] = Evaluation['EndTime'].apply(lambda x: int((float(x.split(':')[0])*60)+float(x.split(':')[1])))

    Evaluation['Start'] = Evaluation['StartTime'].apply(
        lambda x: int((float(x.split(':')[0]) * 4) + float(x.split(':')[1]) // 15))
    Evaluation['End'] = Evaluation['EndTime'].apply(
        lambda x: int((float(x.split(':')[0]) * 4) + float(x.split(':')[1]) // 15))

    Evaluation['Start_End'] = Evaluation.apply(lambda row: StartEnd(row['Start'], row['End']), axis=1)
    Evaluation = Evaluation[['Timestamp', 'Start_End']]
    Evaluation = Evaluation.explode('Start_End')
    Evaluation.drop_duplicates(keep='first', inplace=True)

    anamaly_path = os.path.join(PATH, 'Analysis_PM1', 'Evaluation_Data.csv')
    # Evaluation.to_csv(anamaly_path)
    np.savetxt(anamaly_path, Evaluation, delimiter=',', fmt='%d')
    Common_Row = pd.merge(Evaluation, analomy_data, how='inner', on=['Timestamp', 'Start_End'])
    print('Common_Row', Common_Row.shape[0])
    print('Evaluation_Row', Evaluation.shape[0])
    print('analomy_data_Row', analomy_data.shape[0])
    # compare = datacompy.Compare(Evaluation,analomy_data,join_columns='Timestamp')
    # print(compare.report())
    print('DONE')


def StartEnd(start, end):
    item_list = []
    for item in range(start, end + 1):
        item_list.append(item)
    return item_list


if __name__ == '__main__':
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    folder = 'Analysis_PM1'
    file_name = 'CSVs_datalogger630094-CUBIC_IAQ_PM1_UpToAug22_GT.txt'
    analomy_file = 'anomaly_data(GMT)630094_Per15_Weekall_2021-02-28_2021-08-22.csv'
    analomy_data = pd.read_csv(os.path.join(PATH, folder, analomy_file), usecols=range(0, 2))
    file_path = os.path.join(PATH, folder, file_name)
    Evaluation = read_txt(file_path)
    Evaluation_Process(PATH, Evaluation, analomy_data)
