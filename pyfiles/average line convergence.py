# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:22:30 2021

@author: wuyi1234
"""
# dir_path should contain all the data needed for plotting
dir_path = r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa'

import datetime
import os

import pandas as pd

starting_hour = 16  # which is 24-hour format
execute_once = True

# first step is to find the time interval which files cover through
for file in os.listdir(dir_path):
    CurrFile = pd.read_csv(os.path.join(dir_path, file))
    # drop the last three digits in date (UTC)
    CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
    # convert your timestamps to datetime and then use matplotlib
    CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
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
        # GMT to local time
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

import numpy as np


def moving_average(df, window_size, time_threshold):
    # add nan value for the first few values
    new_col = []
    new_col.extend([np.nan] * (window_size - 1))

    for i in range(0, len(df) - window_size + 1):
        # time interval check
        delta_seconds = (df.iloc[i + window_size - 1, 7] - df.iloc[i, 7]).total_seconds()
        if 0 < delta_seconds <= time_threshold:
            avg = df.iloc[i:(i + window_size), 6].mean()
            new_col.append(avg)
        else:
            new_col.append(np.nan)

    df['ma'] = new_col
    return


import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def accumlating_plot(file_list, date_list, unit_name, save_path):
    weekday_lines = []
    weekend_lines = []
    weekday_index = []
    weekend_index = []

    for i in range(len(date_list)):
        if date_list[i] == 'weekday':
            weekday_index.append(i)
        if date_list[i] == 'weekend':
            weekend_index.append(i)

    for index in weekday_index:
        if len(file_list[index]) > 360:  # ensure that we have enough points for calculation
            # calculate the moving average
            moving_average(file_list[index], 360, 1920)

            # change year,month,day to the same
            mindate = time_list[index][0].date()
            # maxdate=max(CurrFile['date-format']).date())
            file_list[index]['redefined-timestamp'] = file_list[index]['date-format'].map(
                lambda x: x.replace(year=2000, month=1, day=1)
                if x.date() == mindate else x.replace(year=2000, month=1, day=2))
            weekday_lines.append(file_list[index][['date-format', 'redefined-timestamp', 'ma']])

    for index in weekend_index:
        if len(file_list[index]) > 360:
            # calculate the moving average
            moving_average(file_list[index], 360, 1920)
            # change year,month,day to the same
            mindate = time_list[index][0].date()
            # maxdate=max(CurrFile['date-format']).date())
            file_list[index]['redefined-timestamp'] = file_list[index]['date-format'].map(
                lambda x: x.replace(year=2000, month=1, day=1)
                if x.date() == mindate else x.replace(year=2000, month=1, day=2))
            weekend_lines.append(file_list[index][['date-format', 'redefined-timestamp', 'ma']])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 9))

    for line in weekday_lines:
        ax1.plot(line["redefined-timestamp"], line['ma'], 'm')
        print(1)

    for line in weekend_lines:
        ax2.plot(line["redefined-timestamp"], line['ma'], 'm')
        print(2)

    hrlocator = mdates.HourLocator()
    majorFmt = mdates.DateFormatter('%H:%M')

    ax1.xaxis.set_major_locator(hrlocator)
    ax1.xaxis.set_major_formatter(majorFmt)

    ax2.xaxis.set_major_locator(hrlocator)
    ax2.xaxis.set_major_formatter(majorFmt)
    # rotate 90 degrees
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    ax1.set_ylabel('weekday-CO2', fontsize=13)
    ax2.set_ylabel('weekend-CO2', fontsize=13)

    # save prior to show
    file_name = unit_name + ".svg"
    plt.savefig(os.path.join(save_path, file_name), format='svg', dpi=1200, bbox_inches='tight')
    plt.show()

    return


accumlating_plot(file_list, date_list, '61a3fa-CO2', r'C:\Users\wuyi1234\Desktop\New')
##average the line
from sklearn.linear_model import LinearRegression


def exponential_weighting(data_list, factor, biased_correction):
    if biased_correction:
        S = 0
        for i in range(len(data_list)):
            # print(S)
            S = ((1 - factor) * S + factor * data_list[i]) / (1 - (1 - factor) ** (i + 1))

    else:
        # intialize the results
        S = data_list[0]
        for i in range(1, len(data_list)):
            # print(S)
            S = factor * data_list[i] + (1 - factor) * S

    return S


# test
# exponential_weighting([1,2,3],0.9,True)
# exponential_weighting([1,2,3],0.9,False)


def average_line(multiple_line, variable):
    """
    multiple_line should be a list of lines, in which each line should contain a series of point
    """
    reg = LinearRegression()
    only_for_once = True
    for line in multiple_line:
        line.dropna(subset=['redefined-timestamp', 'ma'], inplace=True)
        line['date_ordinal'] = np.arange(1, len(line['redefined-timestamp']) + 1)
        if only_for_once:
            min_time = min(line['redefined-timestamp'])
            max_time = max(line['redefined-timestamp'])
            only_for_once = False

        if min(line['redefined-timestamp']) < min_time:
            min_time = min(line['redefined-timestamp'])
        if max(line['redefined-timestamp']) > max_time:
            max_time = max(line['redefined-timestamp'])
    # generate artificial x value(datetime) sequence
    x_ts = np.arange(min_time, max_time, datetime.timedelta(minutes=5)).astype(datetime.datetime)
    y_ts = np.zeros(shape=(len(x_ts), len(multiple_line)))

    # then find the two nearst timestamp
    i = 0
    for x in x_ts:
        y_avg = []
        for line in multiple_line:
            # find the nearest to the x
            print(x)
            # reset index in case of non-consecutive index when slicing row acorrding to index
            line.reset_index(drop=True, inplace=True)
            line['abs_diff'] = (line['redefined-timestamp'] - x).map(lambda diff: abs(diff.total_seconds()))
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
        # use the exponentially weighting
        # y_ts.append(np.mean(y_avg))
        y_ts[i] = y_avg
        i = i + 1

    results = {}
    # calculate the std everytime
    for j in range(len(multiple_line)):
        y_sub = y_ts[:, 0:(j + 1)]
        results[(j + 1)] = np.apply_along_axis(exponential_weighting, 1, y_sub, factor=0.1, biased_correction=True)

    return x_ts, y_ts, results


from itertools import compress

weekday_lines = list(compress(file_list, np.array(date_list) == "weekday"))
weekend_lines = list(compress(file_list, np.array(date_list) == "weekend"))
# only consider the first 20 lines
x_ts, y_ts, results = average_line(weekday_lines[0:20], 'ma')

for i in range(len(results)):
    plt.figure(figsize=(15, 4))
    # mark the current added line in different color
    ax = plt.gca()

    if i != 0:  # when there is one line at the beginning
        for j in range(i):
            # history data
            if j == 0:
                plt.plot(x_ts, y_ts[:, j], color='green', label='history line(s)')
            else:
                plt.plot(x_ts, y_ts[:, j], color='green')

    # current added line
    plt.plot(x_ts, y_ts[:, i], color='blue', label='current line')
    # average line
    plt.plot(x_ts, results[i + 1], color='red', label='average line')

    hrlocator = mdates.HourLocator()
    majorFmt = mdates.DateFormatter('%H:%M')

    ax.xaxis.set_major_locator(hrlocator)
    ax.xaxis.set_major_formatter(majorFmt)
    # rotate 90 degrees
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.title("Factor: 0.1, average line std is "
              + str(round(np.std(results[i + 1]), 3)) + " After current line is added to the history line(s)")
    ax.set_ylabel('weekday-CO2', fontsize=13)
    plt.legend()
    plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\avg', str(i) + ".svg"), format='svg', dpi=1200,
                bbox_inches='tight')
    plt.show()

# plot the std values of avg line
std_values = []
for i in range(len(results)):
    std_values.append(round(np.std(results[i + 1]), 3))
std_values = np.array(std_values)

plt.plot(range(len(std_values)), std_values)
plt.xticks(range(len(std_values)))
plt.title("std of average line")
plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\avg', "std.svg"), format='svg', dpi=1200, bbox_inches='tight')
plt.show()

# plot the std of the current line
std_values = []
for i in range(y_ts.shape[1]):
    std_values.append(round(np.std(y_ts[:, i]), 3))
std_values = np.array(std_values)
plt.plot(range(len(std_values)), std_values)
plt.xticks(range(len(std_values)))
plt.title("std of current line")
plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\avg', "std.svg"), format='svg', dpi=1200, bbox_inches='tight')
plt.show()

# plot the linear interpolation
# show the difference
for i in range(y_ts.shape[1]):
    plt.plot(x_ts, y_ts[:, i], color='blue', label='linear interpolation ')
    plt.plot(weekday_lines[i]['redefined-timestamp'], weekday_lines[i]['ma'], color="red", label="original line")
    plt.legend()
    plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\comp', str(i) + ".png"), format='png', dpi=600,
                bbox_inches='tight')
    plt.show()
