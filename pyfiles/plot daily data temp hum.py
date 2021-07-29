# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:31:25 2021

@author: wuyi1234
"""

file_path = r'C:\Users\wuyi1234\Desktop\DataLogger\indoor sensor\EcobeeData\Sotiri\Hovardas Ecobee; 2021-01-09-to-2021-02-09.xlsx'

import datetime

import pandas as pd


# def date_time_generator(row):
#     year=row['Date'].year
#     month=row['Date'].month
#     day=row['Date'].day

#     hour=row['Time'].hour
#     minute=row['Time'].minute
#     second=row['Time'].second
#     return datetime.datetime(year,month,day,hour,minute,second)

def date_time_generator(row):
    year = int(row['Date'].split('/')[2])
    month = int(row['Date'].split('/')[0])
    day = int(row['Date'].split('/')[1])

    hour = row['Time'].hour
    minute = row['Time'].minute
    second = row['Time'].second
    return datetime.datetime(year, month, day, hour, minute, second)


CurrFile = pd.read_excel(file_path, skiprows=5)

# extract the key columns
CurrFile = CurrFile[['Date', 'Time', 'Thermostat Temperature (F)', 'Thermostat Humidity (%RH)']]

# clean the data
CurrFile.dropna(axis=0, inplace=True)

# create datetime variable
CurrFile['date-format'] = CurrFile.apply(lambda row: date_time_generator(row), axis=1)

starting_hour = 11  # it is 24-hour format

min_time = min(CurrFile['date-format'])
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
    filtered_file = CurrFile[(starting_date <= CurrFile["date-format"]) &
                             (CurrFile["date-format"] < ending_date)]
    if len(filtered_file) != 0:
        filtered_file = filtered_file.sort_values(by='date-format', ascending=True)
        # remove duplicate rows if there is
        filtered_file = filtered_file.drop_duplicates()
        file_list.append(filtered_file)

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
        delta_seconds = (df.iloc[i + window_size - 1, 4] - df.iloc[i, 4]).total_seconds()
        if 0 < delta_seconds <= time_threshold:
            avg = df.iloc[i:(i + window_size), 3].mean()
            new_col.append(avg)
        else:
            new_col.append(np.nan)

    df['ma'] = new_col
    return


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


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
        if len(file_list[index]) > 6:
            # calculate the moving average
            moving_average(file_list[index], 6, 1920)

            # change year,month,day to the same
            mindate = time_list[index][0].date()
            # maxdate=max(CurrFile['date-format']).date())
            file_list[index]['redefined-timestamp'] = file_list[index]['date-format'].map(
                lambda x: x.replace(year=2000, month=1, day=1)
                if x.date() == mindate else x.replace(year=2000, month=1, day=2))
            weekday_lines.append(file_list[index][['date-format', 'redefined-timestamp', 'ma']])
            print(1)

    for index in weekend_index:
        if len(file_list[index]) > 6:
            # calculate the moving average
            moving_average(file_list[index], 6, 1920)
            # change year,month,day to the same
            mindate = time_list[index][0].date()
            # maxdate=max(CurrFile['date-format']).date())
            file_list[index]['redefined-timestamp'] = file_list[index]['date-format'].map(
                lambda x: x.replace(year=2000, month=1, day=1)
                if x.date() == mindate else x.replace(year=2000, month=1, day=2))
            weekend_lines.append(file_list[index][['date-format', 'redefined-timestamp', 'ma']])
            print(2)

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
    ax1.set_ylabel('weekday-humidity', fontsize=13)
    ax2.set_ylabel('weekend-humidity', fontsize=13)

    # save prior to show
    file_name = unit_name + ".svg"
    plt.savefig(os.path.join(save_path, file_name), format='svg', dpi=1200, bbox_inches='tight')

    plt.show()

    return


accumlating_plot(file_list, date_list, '630094-humidity', r'C:\Users\wuyi1234\Desktop\New')
