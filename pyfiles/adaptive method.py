# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:47:06 2021

@author: WuYi1234
"""

# dir_path should contain all the data needed for plotting
# dir_path=r'C:\Users\wuyi1234\Desktop\IEQ\FileCollection'

import datetime
import os
from itertools import compress

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# dir_path should contain all the data needed for plotting
PATH = os.path.abspath(os.path.dirname(os.getcwd()))
dir_path = PATH + r'\CO2_DATA'
print('Data path is:', dir_path)

starting_hour = 16  # which is 24-hour format
execute_once = True


def data_prepro():
    # first step is to find the time interval which files cover through
    for file in os.listdir(dir_path):
        CurrFile = pd.read_csv(os.path.join(dir_path, file))
        # drop the last three digits in date (UTC)
        CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        CurrFile["date-format"] = CurrFile["date (UTC)"].map(
            lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # sort timestamp in ascending order
        CurrFile = CurrFile.sort_values(by='date-format', ascending=True)
        # delete the NA rows
        CurrFile = CurrFile.dropna()

        if execute_once:
            min_time = min(CurrFile['date-format'])
            max_time = max(CurrFile['date-format'])
            execute_once = False

        if min(CurrFile['date-format']) < min_time:
            min_time = min(CurrFile['date-format'])
        if max(CurrFile['date-format']) > max_time:
            max_time = max(CurrFile['date-format'])

    # each element in file_list would be one day-data
    file_list = []
    # each element in date_list would be corresponding to weekday of the 24-hour
    # len(date_list) should be equal to the file_list
    date_list = []
    time_list = []

    starting_date = pd.Timestamp(min_time.year, min_time.month, min_time.day,
                                 starting_hour, 0, 0)

    ending_date = starting_date + pd.Timedelta(days=1)

    while ending_date < max_time + pd.Timedelta(days=1):
        temp_df = pd.DataFrame()
        for file in os.listdir(dir_path):  # since 24-hour data may be contained in two files or more
            CurrFile = pd.read_csv(os.path.join(dir_path, file))
            # drop the last three digits in date (UTC)
            CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
            # convert your timestamps to datetime and then use matplotlib
            CurrFile["date-format"] = CurrFile["date (UTC)"].map(
                lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
            # sort timestamp in ascending order
            CurrFile = CurrFile.sort_values(by='date-format', ascending=True)
            # delete the NA rows
            CurrFile = CurrFile.dropna()

            filtered_file = CurrFile[(starting_date <= CurrFile["date-format"]) &
                                     (CurrFile["date-format"] < ending_date)]
            if len(filtered_file) != 0:
                temp_df = temp_df.append(filtered_file)

        if len(temp_df) != 0:  # there could be some missing day
            temp_df = temp_df.sort_values(by='date-format', ascending=True)
            # remove duplicate rows if there is
            temp_df = temp_df.drop_duplicates()
            # GMT to local time(GMT- 5 hours)
            temp_df['date-format'] = temp_df['date-format'] - pd.Timedelta(hours=5)
            file_list.append(temp_df)

            if starting_date.weekday() in [0, 1, 2, 3]:
                date_list.append('weekday')
            else:
                date_list.append('weekend')

            time_list.append([starting_date, ending_date])

        starting_date += pd.Timedelta(days=1)
        ending_date += pd.Timedelta(days=1)

        print(starting_date)
        print(ending_date)

    # add redefined timestamp column
    i = 0
    for starting_date, ending_date in time_list:
        print(str(starting_date) + '    ' + str(ending_date))
        mindate = starting_date.date()
        file_list[i]['redefined-timestamp'] = file_list[i]['date-format'].map(
            lambda x: x.replace(year=2000, month=1, day=1)
            if x.date() == mindate else x.replace(year=2000, month=1, day=2))
        file_list[i]['date_ordinal'] = np.arange(1, len(file_list[i]) + 1)
        i = i + 1
    return file_list, date_list, time_list


# ///// 数据处理，24小时数据


def data_visualization(file_list):
    # visualize the results
    plt.figure(figsize=(15, 4))
    # mark the current added line in different color
    ax = plt.gca()

    for i in range(len(file_list)):
        # history line
        plt.plot(file_list[i]['redefined-timestamp'],
                 file_list[i]['value'], color='m')

    hrlocator = mdates.HourLocator()
    majorFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(hrlocator)
    ax.xaxis.set_major_formatter(majorFmt)
    # rotate 90 degrees
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.show()

    # plot all the data, 单位为24小时


# test by Aaron--------------------------------------------------------------------------------------------------------------
def FindOrInterpolation(x_ts, multiple_line, variable):
    """
    multiple_line should be a list of lines, in which each line should contain a series of point
    This anomaly detection is based on the history data at the same time
    """
    reg = LinearRegression()

    # x_ts=np.arange(min_time,max_time,datetime.timedelta(minutes=5)).astype(datetime.datetime)
    y_ts = np.zeros(shape=(len(x_ts), len(multiple_line)))

    # then find the two nearst timestamp
    i = 0
    for x in x_ts:
        y_avg = []
        for line in multiple_line:
            # find the nearest to the x
            # print(x)
            # reset index in case of non-consecutive index when slicing row acorrding to index
            line.reset_index(drop=True, inplace=True)
            line['abs_diff'] = (line['redefined-timestamp'] - x).map(lambda diff: abs(diff.total_seconds()))
            # find the nearest neighbor location
            min_index = line['abs_diff'].idxmin()
            if line.loc[min_index]['abs_diff'] == 0:
                y_avg.append(line.loc[min_index][variable])
            else:
                if min_index == min(line.index):
                    two_points = line.loc[[min_index, min_index + 1]]

                elif min_index == max(line.index):
                    two_points = line.loc[[min_index, min_index - 1]]


                else:
                    temp_df = line.loc[[min_index - 1, min_index + 1]]
                    second_min_index = temp_df['abs_diff'].idxmin()
                    two_points = line.loc[[min_index, second_min_index]]

                arr = np.array(two_points['date_ordinal'])
                arr = arr.reshape(-1, 1)
                # fit linear regression
                reg.fit(arr, two_points[variable])
                # then predict
                # transform the x value into ordinal scale
                time_diff = abs((two_points.iloc[0]['redefined-timestamp'] - two_points.iloc[1][
                    'redefined-timestamp']).total_seconds())

                if x > line.loc[min_index]['redefined-timestamp']:
                    x_temp = line.loc[min_index]['date_ordinal'] + (line.loc[min_index]['abs_diff'] / time_diff) * abs(
                        two_points.iloc[0]['date_ordinal'] - two_points.iloc[1]['date_ordinal'])

                elif x < line.loc[min_index]['redefined-timestamp']:
                    x_temp = line.loc[min_index]['date_ordinal'] - (line.loc[min_index]['abs_diff'] / time_diff) * abs(
                        two_points.iloc[0]['date_ordinal'] - two_points.iloc[1]['date_ordinal'])

                x_temp = np.array(x_temp)
                x_temp = x_temp.reshape(-1, 1)

                y_pred = reg.predict(x_temp)
                y_avg.append(y_pred[0])
        # use the avg
        y_ts[i] = y_avg
        i = i + 1

    return y_ts


def AnomalyDetection(multiple_line):
    """
    
    Parameters
    ----------
    timestamp : Timestamp
        the target for to look for 
    multiple_line : List
        a list of daily data

    Returns
    -------
    a list of values at the same time

    """
    anomaly = {}

    j = 0
    for line in multiple_line:
        print(j)
        x_ts = line['redefined-timestamp']
        y_ts = FindOrInterpolation(x_ts, multiple_line, 'value')
        # then judge whether it is anomly or not for the current line
        for i in range(y_ts.shape[0]):
            mean = np.mean(y_ts[i])
            std = np.std(y_ts[i])
            upper_bound = mean + 2 * std
            lower_bound = mean - 2 * std
            if line['value'][i] < lower_bound or line['value'][i] > upper_bound:
                # add it to the anomaly list
                if j in anomaly.keys():
                    anomaly[j] = anomaly[j].append(line.iloc[i])
                else:
                    anomaly[j] = line.iloc[i]
        j = j + 1


# AnomalyDetection(file_list)


def AnomalyDetection(multiple_line, interval_in_seconds=60):
    # find the min time and max time of multiple lines
    only_for_once = True
    for line in multiple_line:
        line.dropna(subset=['redefined-timestamp', 'value'], inplace=True)
        if only_for_once:
            min_time = min(line['redefined-timestamp'])
            max_time = max(line['redefined-timestamp'])
            only_for_once = False

        if min(line['redefined-timestamp']) < min_time:
            min_time = min(line['redefined-timestamp'])
        if max(line['redefined-timestamp']) > max_time:
            max_time = max(line['redefined-timestamp'])

    interval = pd.Timedelta(seconds=interval_in_seconds)

    anomaly_dict = {}
    limit = []
    while min_time < max_time:
        # select out the data within this time bin
        print(min_time)
        temp_df = pd.DataFrame()
        temp_df_dict = {}
        i = 0
        for line in multiple_line:
            filtered_file = line[(min_time <= line['redefined-timestamp']) &
                                 (line['redefined-timestamp'] < (min_time + interval))]
            if len(filtered_file) != 0:
                temp_df = temp_df.append(filtered_file)
                temp_df_dict[i] = filtered_file
            i = i + 1

        # calculate the mean and std within the time bin
        bin_mean = np.mean(temp_df['value'])
        bin_std = np.std(temp_df['value'])
        upper_bound = bin_mean + 2 * bin_std
        lower_bound = bin_mean - 2 * bin_std

        limit.append((min_time, bin_mean, bin_std))
        # flag the anomaly
        for key in temp_df_dict.keys():
            anomaly = temp_df_dict[key][(temp_df_dict[key]['value'] < lower_bound) |
                                        (temp_df_dict[key]['value'] > upper_bound)]
            if len(anomaly) != 0:
                if key in anomaly_dict.keys():
                    anomaly_dict[key] = anomaly_dict[key].append(anomaly)
                else:
                    anomaly_dict[key] = anomaly

        min_time = min_time + interval
    return anomaly_dict, limit


# anomaly_points,limit=AnomalyDetection(file_list)

# plot
# plt.figure(figsize=(15, 4))
# #mark the current added line in different color
# ax = plt.gca()

# #anamaly part
# for key in anomaly_points.keys():
#     #plot
#     plt.figure(figsize=(15, 4))
#     #mark the current added line in different color
#     ax = plt.gca()

#     plt.scatter(anomaly_points[key]['redefined-timestamp'],
#                 anomaly_points[key]['value'],color='r',s=10,marker='X')
#     plt.plot(file_list[key]['redefined-timestamp'],
#              file_list[key]['value'],color='green')
#     plt.title(str(time_list[key][0])+'-'+str(time_list[key][1]))

#     hrlocator=mdates.HourLocator()
#     majorFmt = mdates.DateFormatter('%H:%M') 
#     ax.xaxis.set_major_locator(hrlocator)
#     ax.xaxis.set_major_formatter(majorFmt)    
#     #rotate 90 degrees
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)     
#     plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\New',str(key)+".png"),format='png', dpi=600,bbox_inches = 'tight')
#     plt.show()


# #ribbon
# plt.figure(figsize=(15, 4))
# #mark the current added line in different color
# ax = plt.gca()
# for time_slot in limit:
#     plt.hlines(y=time_slot[1],
#                xmin=time_slot[0],xmax=time_slot[0]+pd.Timedelta(seconds=60),
#                color='red')
#     # ax.fill_between([time_slot[0],time_slot[0]+pd.Timedelta(seconds=60)],
#     #                 time_slot[1]-2*time_slot[2],
#     #                 time_slot[1]+2*time_slot[2],
#     #                 color='crimson',alpha=0.1)

# hrlocator=mdates.HourLocator()
# majorFmt = mdates.DateFormatter('%H:%M') 
# ax.xaxis.set_major_locator(hrlocator)
# ax.xaxis.set_major_formatter(majorFmt)    
# #rotate 90 degrees
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)       
# plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\New',"aaa.svg"),format='svg', dpi=600,bbox_inches = 'tight')
# plt.show()


# #supercomposed together
# for key in anomaly_points.keys():
#     #plot
#     plt.figure(figsize=(15, 4))
#     #mark the current added line in different color
#     ax = plt.gca()

#     for time_slot in limit:
#         plt.hlines(y=time_slot[1],
#                xmin=time_slot[0],xmax=time_slot[0]+pd.Timedelta(seconds=60),
#                color='red')
#         ax.fill_between([time_slot[0],time_slot[0]+pd.Timedelta(seconds=60)],
#                     time_slot[1]-2*time_slot[2],
#                     time_slot[1]+2*time_slot[2],
#                     color='crimson',alpha=0.1)

#     plt.scatter(anomaly_points[key]['redefined-timestamp'],
#                 anomaly_points[key]['value'],color='r',s=10,marker='X')
#     plt.plot(file_list[key]['redefined-timestamp'],
#              file_list[key]['value'],color='green')
#     plt.title(str(time_list[key][0])+'-'+str(time_list[key][1]))

#     hrlocator=mdates.HourLocator()
#     majorFmt = mdates.DateFormatter('%H:%M') 
#     ax.xaxis.set_major_locator(hrlocator)
#     ax.xaxis.set_major_formatter(majorFmt)    
#     #rotate 90 degrees
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)     
#     plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\New',str(key)+".png"),format='png', dpi=600,bbox_inches = 'tight')
#     plt.show()

#######################################################################

def AdaptiveAnomalyDetection(multiple_line, interval_in_seconds=60, baseline_day=1):
    # first multiple lines contain line in chronological order
    # add each line a unique number to indicate the order of line
    # this would prevent the situation when there will be some missing data within one bin
    multiple_line_dict = {}
    for j in range(len(multiple_line)):
        multiple_line_dict[j] = multiple_line[j]

    visit_multi_seq = list(multiple_line_dict.keys())
    visit_multi_seq.sort()

    # find the min time and max time of multiple lines
    only_for_once = True
    for line in multiple_line:
        line.dropna(subset=['redefined-timestamp', 'value'], inplace=True)
        if only_for_once:
            min_time = min(line['redefined-timestamp'])
            max_time = max(line['redefined-timestamp'])
            only_for_once = False

        if min(line['redefined-timestamp']) < min_time:
            min_time = min(line['redefined-timestamp'])
        if max(line['redefined-timestamp']) > max_time:
            max_time = max(line['redefined-timestamp'])
    # the non-overlapping window length
    interval = pd.Timedelta(seconds=interval_in_seconds)

    anomaly_dict = {}
    normal_dict = []  # dict-like list
    limit = []
    bin_counter = 0  # bin0, bin1, bin2

    while min_time < max_time:
        # select out the data within this time bin
        print(min_time)
        temp_df_dict = {}
        # record the normal data
        normal = []

        for line_key in visit_multi_seq:
            line = multiple_line_dict[line_key]
            filtered_file = line[(min_time <= line['redefined-timestamp']) &
                                 (line['redefined-timestamp'] < (min_time + interval))]
            if len(filtered_file) != 0:
                temp_df_dict[line_key] = filtered_file.copy()

        # calculate the mean and std in adaptive way
        keys = list(temp_df_dict.keys())
        # in increasing order and will add them one by one
        keys.sort()

        # calculate the first few slots as baseline(e.g. Day-1 data)
        baseline_keys = keys[0:baseline_day]
        for baseline_key in baseline_keys:
            normal.extend(temp_df_dict[baseline_key]['value'])
            normal_dict.append([bin_counter, min_time, baseline_key, normal.copy()])

        for key in keys:
            lower_bound = np.mean(normal) - 2 * np.std(normal)
            upper_bound = np.mean(normal) + 2 * np.std(normal)

            if key not in baseline_keys:
                anomaly = temp_df_dict[key][(temp_df_dict[key]['value'] < lower_bound) |
                                            (temp_df_dict[key]['value'] > upper_bound)]

                if len(anomaly) != 0:
                    # check whether one part of line is already flagged as anomaly
                    if key in anomaly_dict.keys():
                        anomaly_dict[key] = anomaly_dict[key].append(anomaly)
                        limit.append((min_time, key, np.mean(normal), np.std(normal)))

                    else:
                        anomaly_dict[key] = anomaly
                        limit.append((min_time, key, np.mean(normal), np.std(normal)))

                # normal data
                temp_normal = temp_df_dict[key][(temp_df_dict[key]['value'] >= lower_bound) &
                                                (temp_df_dict[key]['value'] <= upper_bound)]

                if len(temp_normal) != 0:
                    # then add new data into the normal group
                    normal.extend(temp_normal['value'])

                normal_dict.append([bin_counter, min_time, key, normal.copy()])

        min_time = min_time + interval
        bin_counter = bin_counter + 1
    return anomaly_dict, limit, normal_dict, bin_counter


# test by Aaron----------------------------------------------------------------------------------------------------------------------
# anomaly_points,limit,normal_dict,bin_counter=AdaptiveAnomalyDetection(file_list,baseline_day=1)    


# supercomposed together
# for key in anomaly_points.keys():
#     print(key)
#     #plot
#     plt.figure(figsize=(15, 4))
#     #mark the current added line in different color
#     ax = plt.gca()

#     bins={}
#     #find the corresponding value in each bin
#     for normal_slot in normal_dict:
#         bin_num=normal_slot[0]
#         #find the nearest bin(<= key)
#         if bin_num not in bins.keys() and normal_slot[2]<key:
#             bins[bin_num]=normal_slot
#         elif bin_num in bins.keys():
#             if normal_slot[2]<key and normal_slot[2]>bins[bin_num][2]:
#                 bins[bin_num]=normal_slot

#     #draw the corresponding ribbon first
#     for each_bin_index in bins.keys():
#         each_bin=bins[each_bin_index]
#         plt.hlines(y=np.mean(each_bin[3]),
#                        xmin=each_bin[1],xmax=each_bin[1]+pd.Timedelta(seconds=60),
#                        color='red')
#         ax.fill_between([each_bin[1],each_bin[1]+pd.Timedelta(seconds=60)],
#                             np.mean(each_bin[3])-2*np.std(each_bin[3]), 
#                             np.mean(each_bin[3])+2*np.std(each_bin[3]), 
#                             color='crimson',alpha=0.1)

#     plt.scatter(anomaly_points[key]['redefined-timestamp'],
#                 anomaly_points[key]['value'],color='r',s=10,marker='X')
#     plt.plot(file_list[key]['redefined-timestamp'],
#              file_list[key]['value'],color='green')
#     plt.title(str(time_list[key][0])+'-'+str(time_list[key][1]))

#     hrlocator=mdates.HourLocator()
#     majorFmt = mdates.DateFormatter('%H:%M') 
#     ax.xaxis.set_major_locator(hrlocator)
#     ax.xaxis.set_major_formatter(majorFmt)    
#     #rotate 90 degrees
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)     
#     plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\NEW3',str(key)+".png"),format='png', dpi=600,bbox_inches = 'tight')
#     plt.show()


# Action
# 1. using 1-minute average data as a single-point data for a "bin"
# For every day, the data vector is of length 60x24 = 1440

# 2. Try different std mulitipication factor (2, or 3, or 2.5)

# 3. Implement the above-mentioned "ribbon"-update scheme

# 4. Use "connected bins" to pick out significantly long abnormal events.

# 每一个bin处理完才进行下一个bin
def M_AdaptiveAnomalyDetection(multiple_line, mulitipication_factor, ribbon_factor, interval_in_seconds=60,
                               baseline_day=1):
    # first multiple lines contain line in chronological order
    # add each line a unique number to indicate the order of line
    # this would prevent the situation when there will be some missing data within one bin
    multiple_line_dict = {}
    for j in range(len(multiple_line)):
        multiple_line_dict[j] = multiple_line[j]

    visit_multi_seq = list(multiple_line_dict.keys())
    visit_multi_seq.sort()

    # find the min time and max time of multiple lines
    only_for_once = True
    for line in multiple_line:
        line.dropna(subset=['redefined-timestamp', 'value'], inplace=True)
        if only_for_once:
            min_time = min(line['redefined-timestamp'])
            max_time = max(line['redefined-timestamp'])
            only_for_once = False

        if min(line['redefined-timestamp']) < min_time:
            min_time = min(line['redefined-timestamp'])
        if max(line['redefined-timestamp']) > max_time:
            max_time = max(line['redefined-timestamp'])
    # the non-overlapping window length
    interval = pd.Timedelta(seconds=interval_in_seconds)

    anomaly_dict = {}
    normal_dict = []  # dict-like list
    limit = []
    bin_dict = {}  # each bin will contain multiple (part)lines
    bin_counter = 0  # bin0, bin1, bin2

    while min_time < max_time:
        # select out the data within this time bin
        print(min_time)
        temp_df_dict = {}
        # record the normal data
        normal = []

        for line_key in visit_multi_seq:
            line = multiple_line_dict[line_key]
            filtered_file = line[(min_time <= line['redefined-timestamp']) &
                                 (line['redefined-timestamp'] < (min_time + interval))]
            if len(filtered_file) != 0:
                temp_df_dict[line_key] = filtered_file.copy()
                # then add to the bin_dict(for drawing the plot)
                if bin_counter not in bin_dict.keys():
                    bin_dict[bin_counter] = [(line_key, min_time, filtered_file.copy()['value'])]
                else:
                    bin_dict[bin_counter].append((line_key, min_time, filtered_file.copy()['value']))

        # calculate the mean and std in adaptive way
        keys = list(temp_df_dict.keys())
        # in increasing order and will add them one by one
        keys.sort()

        # use the average value in 1-min non-overlapping window
        for line_key in keys:
            temp_df_dict[line_key] = np.mean(temp_df_dict[line_key]['value'])

        # calculate the first few days as baseline(e.g. Day-1 data)
        baseline_keys = keys[0:baseline_day]
        for baseline_key in baseline_keys:
            # extract the 1-min average
            normal.append(temp_df_dict[baseline_key])

            normal_dict.append([bin_counter, min_time, baseline_key, normal.copy()])

        for key in keys:
            lower_bound = np.mean(normal) - ribbon_factor * np.std(normal)
            upper_bound = np.mean(normal) + ribbon_factor * np.std(normal)

            if key not in baseline_keys:

                if temp_df_dict[key] < lower_bound or temp_df_dict[key] > upper_bound:
                    # then it is anomaly
                    if key in anomaly_dict.keys():
                        anomaly_dict[key].append((min_time, temp_df_dict[key]))

                    else:
                        anomaly_dict[key] = [(min_time, temp_df_dict[key])]

                    limit.append((min_time, key, np.mean(normal), np.std(normal)))

                    # then it still has impact on the calculating the means and std controlled by mulitipication_factor
                    normal.append(temp_df_dict[key] * mulitipication_factor)

                elif lower_bound <= temp_df_dict[key] and temp_df_dict[key] <= upper_bound:
                    # then it is normal and add it to normal group
                    normal.append(temp_df_dict[key])

                normal_dict.append([bin_counter, min_time, key, normal.copy()])

        min_time = min_time + interval
        bin_counter = bin_counter + 1
    return anomaly_dict, limit, normal_dict, bin_dict, bin_counter


# only consecutive anomaly points >= consecutive_bin_size, then it will be recorded

# 异常持续<10min(consecutive_bin_size),认为是正常值
def bin_connected(consecutive_bin_size, anomaly_points_dict):
    consecutive_indicator_dict = {}

    for line_key in anomaly_points_dict:
        # for each line, it may include many non-consecutive segments
        anomaly_points_each_line = anomaly_points_dict[line_key]
        consecutive_indicator = np.zeros(shape=(len(anomaly_points_each_line)), dtype=int)
        # initialization
        group_num = 1
        consecutive_indicator[0] = 1
        starting_time = anomaly_points_each_line[0][0]

        for i in range(1, len(anomaly_points_each_line)):
            curr_anomaly_point = anomaly_points_each_line[i]
            if curr_anomaly_point[0] == starting_time + pd.Timedelta(seconds=60):
                # then it is consecutive
                consecutive_indicator[i] = group_num
                starting_time = starting_time + pd.Timedelta(seconds=60)
            else:
                group_num = group_num + 1
                consecutive_indicator[i] = group_num
                starting_time = curr_anomaly_point[0]

        consecutive_indicator_dict[line_key] = consecutive_indicator
        processed_anomaly_points_dict = {}
        suppressed_anomaly_points_dict = {}
        # then suppress the anamoly which is less than consecutive_bin_size
        for line_key in consecutive_indicator_dict:
            (unique, counts) = np.unique(consecutive_indicator_dict[line_key], return_counts=True)
            suppressed_group = unique[counts < consecutive_bin_size]
            pos_bool_filter = np.ones(shape=len(consecutive_indicator_dict[line_key]), dtype=bool)
            neg_bool_filter = np.zeros(shape=len(consecutive_indicator_dict[line_key]), dtype=bool)
            for group_num in suppressed_group:
                pos_bool_filter[np.where(consecutive_indicator_dict[line_key] == group_num)] = False
                neg_bool_filter[np.where(consecutive_indicator_dict[line_key] == group_num)] = True

            processed_anomaly_points_dict[line_key] = list(compress(anomaly_points_dict[line_key], pos_bool_filter))
            suppressed_anomaly_points_dict[line_key] = list(compress(anomaly_points_dict[line_key], neg_bool_filter))
    # processed_anomaly_points_collection
    return consecutive_indicator_dict, processed_anomaly_points_dict, suppressed_anomaly_points_dict


# ribbon_factor=3
# #supercomposed together
# for key in anomaly_points.keys():
#     print(key)
#     #plot
#     plt.figure(figsize=(15, 4))
#     #mark the current added line in different color
#     ax = plt.gca()

#     bins={} # for drawing the ribbon
#     #find the corresponding value in each bin
#     for normal_slot in normal_dict:
#         bin_num=normal_slot[0]
#         #find the nearest bin(<= key)
#         if bin_num not in bins.keys() and normal_slot[2]<key:
#             bins[bin_num]=normal_slot
#         elif bin_num in bins.keys():
#             if normal_slot[2]<key and normal_slot[2]>bins[bin_num][2]:
#                 bins[bin_num]=normal_slot


#     #draw the corresponding ribbon first
#     for each_bin_index in bins.keys():
#         each_bin=bins[each_bin_index]
#         plt.hlines(y=np.mean(each_bin[3]),
#                         xmin=each_bin[1],xmax=each_bin[1]+pd.Timedelta(seconds=60),
#                         color='crimson')
#         ax.fill_between([each_bin[1],each_bin[1]+pd.Timedelta(seconds=60)],
#                             np.mean(each_bin[3])-ribbon_factor*np.std(each_bin[3]), 
#                             np.mean(each_bin[3])+ribbon_factor*np.std(each_bin[3]), 
#                             color='crimson',alpha=0.1)

#     #draw 1-min average instead of the points       
#     for ith_bin in bin_dict.keys():
#         for line in bin_dict[ith_bin]:
#             if line[0]==key:
#                 plt.hlines(y=np.mean(line[2]),
#                            xmin=line[1],xmax=line[1]+pd.Timedelta(seconds=60),
#                            color='green')

#     #draw 1-min anomaly          
#     for anomaly_segment in anomaly_points[key]:
#         plt.hlines(y=anomaly_segment[1],
#                    xmin=anomaly_segment[0],xmax=anomaly_segment[0]+pd.Timedelta(seconds=60),
#                    linewidth=6.0,color='#9A0EEA')


#     plt.title(str(time_list[key][0]-pd.Timedelta(hours=5))+'-'+str(time_list[key][1]-pd.Timedelta(hours=5)))
#     ax.set_ylim([-250,2000])
#     hrlocator=mdates.HourLocator()
#     majorFmt = mdates.DateFormatter('%H:%M') 
#     ax.xaxis.set_major_locator(hrlocator)
#     ax.xaxis.set_major_formatter(majorFmt)    
#     #rotate 90 degrees
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)     
#     plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\NEW3',str(key)+".png"),format='png', dpi=600,bbox_inches = 'tight')
#     plt.show()


#######################################above is the copy of drawing (without considering bin-connted method)#########################################
# with bin connected suppression method

# supercomposed together

# 开始画图
def plot_PIC():
    for key in anomaly_points.keys():
        print(key)
        # plot
        plt.figure(figsize=(15, 4))
        # mark the current added line in different color
        ax = plt.gca()

        bins = {}  # for drawing the ribbon
        # find the corresponding value in each bin
        for normal_slot in normal_dict:
            bin_num = normal_slot[0]
            # find the nearest bin(<= key)
            if bin_num not in bins.keys() and normal_slot[2] < key:
                bins[bin_num] = normal_slot
            elif bin_num in bins.keys():
                if normal_slot[2] < key and normal_slot[2] > bins[bin_num][2]:
                    bins[bin_num] = normal_slot

        # draw the corresponding ribbon first
        for each_bin_index in bins.keys():
            each_bin = bins[each_bin_index]
            plt.hlines(y=np.mean(each_bin[3]),
                       xmin=each_bin[1], xmax=each_bin[1] + pd.Timedelta(seconds=60),
                       color='crimson')
            ax.fill_between([each_bin[1], each_bin[1] + pd.Timedelta(seconds=60)],
                            np.mean(each_bin[3]) - ribbon_factor * np.std(each_bin[3]),
                            np.mean(each_bin[3]) + ribbon_factor * np.std(each_bin[3]),
                            color='crimson', alpha=0.1)

        # draw 1-min average instead of the points
        for ith_bin in bin_dict.keys():
            for line in bin_dict[ith_bin]:
                if line[0] == key:
                    plt.hlines(y=np.mean(line[2]),
                               xmin=line[1], xmax=line[1] + pd.Timedelta(seconds=60),
                               color='green')

        # draw 1-min long-term anamoly
        for anomaly_segment in processed_anomaly_points_dict[key]:
            plt.hlines(y=anomaly_segment[1],
                       xmin=anomaly_segment[0], xmax=anomaly_segment[0] + pd.Timedelta(seconds=60),
                       linewidth=6.0, color='#9A0EEA')

        # draw suppressed anamoly
        for suppressed_anomaly_segment in suppressed_anomaly_points_dict[key]:
            plt.hlines(y=suppressed_anomaly_segment[1],
                       xmin=suppressed_anomaly_segment[0],
                       xmax=suppressed_anomaly_segment[0] + pd.Timedelta(seconds=60),
                       linewidth=6.0, color='#eadb0e')

        plt.title(str(time_list[key][0] - pd.Timedelta(hours=5)) + '-' + str(time_list[key][1] - pd.Timedelta(hours=5)))
        ax.set_ylim([-250, 2000])
        hrlocator = mdates.HourLocator()
        majorFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_locator(hrlocator)
        ax.xaxis.set_major_formatter(majorFmt)
        # rotate 90 degrees
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        if not os.path.exists(PATH + r'\export_PIC'):
            os.mkdir(PATH + r'\export_PIC')
        plt.savefig(os.path.join(PATH + r'\export_PIC', str(key) + ".png"), format='png', dpi=600, bbox_inches='tight')

        plt.show()


if __name__ == '__main__':
    file_list, time_list, date_list = data_prepro()
    data_visualization(file_list)
    # mulitipication_factor is beta
    ribbon_factor = 3  # is alpha value
    anomaly_points, limit, normal_dict, bin_dict, bin_counter = M_AdaptiveAnomalyDetection(file_list,
                                                                                           mulitipication_factor=0.1,
                                                                                           ribbon_factor=3,
                                                                                           interval_in_seconds=60,
                                                                                           baseline_day=7)

    consecutive_indicator_dict, processed_anomaly_points_dict, suppressed_anomaly_points_dict = bin_connected(10,
                                                                                                              anomaly_points)
    plot_PIC()
