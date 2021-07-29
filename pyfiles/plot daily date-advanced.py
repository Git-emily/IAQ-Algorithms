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
    ax1.set_ylabel('weekday-VOC', fontsize=13)
    ax2.set_ylabel('weekend-VOC', fontsize=13)

    # save prior to show
    file_name = unit_name + ".svg"
    plt.savefig(os.path.join(save_path, file_name), format='svg', dpi=1200, bbox_inches='tight')

    plt.show()

    return


accumlating_plot(file_list, date_list, '630094-VOC', r'C:\Users\wuyi1234\Desktop\New')
