# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:47:06 2021

@author: WuYi1234
"""

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


def data_prepro(): #Every 24h
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
        global execute_once   # modify a global variable, must use the keyword 'global'
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

# Action
# 1. using 1-minute average data as a single-point data for a "bin"
# For every day, the data vector is of length 60x24 = 1440

# 2. Try different std mulitipication factor (2, or 3, or 2.5)

# 3. Implement the above-mentioned "ribbon"-update scheme

# 4. Use "connected bins" to pick out significantly long abnormal events.

# running bin by bin
def M_AdaptiveAnomalyDetection(multiple_line, mulitipication_factor, ribbon_factor, interval_in_seconds=60,baseline_day=1):
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

# plot
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
    ribbon_factor = 3  # is alpha value
    mulitipication_factor = 0.1  # is beta value
    interval_in_seconds = 60
    baseline_day = 7
    anomaly_points, limit, normal_dict, bin_dict, bin_counter = M_AdaptiveAnomalyDetection(file_list,
                                                                                           mulitipication_factor,
                                                                                           ribbon_factor,
                                                                                           interval_in_seconds,
                                                                                           baseline_day)

    consecutive_indicator_dict, processed_anomaly_points_dict, suppressed_anomaly_points_dict = bin_connected(10,
                                                                                                              anomaly_points)
    plot_PIC()
