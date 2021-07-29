# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:05:30 2021

@author: WuYi
"""

import os

import numpy as np
import pandas as pd


def file_searching(folder_path, file_name):
    """
    Parameters
    ----------
    folder_path : the folder path which contains the file you want to search
    file_name : the file you want to search 

    Returns
    -------
    files_list : the list contains the absolute path pointing to the file you want to search

    """
    files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_name in file:
                files_list.append((os.path.join(root, file)))
    return files_list


import datetime


def date_time_generator(row):
    if isinstance(row['Date'], str):

        year = int(row['Date'].split('/')[2])
        month = int(row['Date'].split('/')[0])
        day = int(row['Date'].split('/')[1])

        hour = row['Time'].hour
        minute = row['Time'].minute
        second = row['Time'].second

    else:
        year = row['Date'].year
        month = row['Date'].month
        day = row['Date'].day

        hour = row['Time'].hour
        minute = row['Time'].minute
        second = row['Time'].second

    return datetime.datetime(year, month, day, hour, minute, second)


def IAQIndexCalculation(IndoorTempHumidity_file_path,
                        folder_path,  # which contains the CO2,PM2.5 and VOC files
                        CO2_file_name,
                        PM25_file_name,
                        VOC_file_name):
    # use indoor temperature and humidity as the main entity
    IndoorTempHumidity = pd.read_excel(IndoorTempHumidity_file_path,
                                       skiprows=5)

    # extract the key columns
    IndoorTempHumidity = IndoorTempHumidity[['Date', 'Time', 'Thermostat Temperature (F)', 'Thermostat Humidity (%RH)']]

    # clean the data
    IndoorTempHumidity.dropna(axis=0, inplace=True)

    # create datetime variable
    IndoorTempHumidity['date_time'] = IndoorTempHumidity.apply(lambda row: date_time_generator(row), axis=1)

    # find the relevent file and combine
    starting_date_time = min(IndoorTempHumidity['date_time'])
    ending_date_time = max(IndoorTempHumidity['date_time'])

    # adding prior and after "padding" (extend the original range)
    starting_date_time = starting_date_time - datetime.timedelta(minutes=1)
    ending_date_time = ending_date_time + datetime.timedelta(minutes=1)

    # find the other sensor data
    CO2_files_list = file_searching(folder_path, CO2_file_name)
    PM25_files_list = file_searching(folder_path, PM25_file_name)
    VOC_files_list = file_searching(folder_path, VOC_file_name)

    CO2_info = pd.DataFrame()
    for CO2_file_path in CO2_files_list:
        CO2_pd = pd.read_csv(CO2_file_path)
        # drop the last three digits in date (UTC)
        CO2_pd["date (UTC)"] = CO2_pd["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        CO2_pd["date-format"] = CO2_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

        # drop the empty value
        CO2_pd = CO2_pd[["date-format", "value"]]
        CO2_pd.rename(columns={'value': "CO2"}, inplace=True)
        CO2_pd["date-format"] = CO2_pd["date-format"] - pd.Timedelta(hours=5)
        CO2_pd = CO2_pd.dropna()

        CO2_pd = CO2_pd[(CO2_pd['date-format'] >= starting_date_time) &
                        (CO2_pd['date-format'] <= ending_date_time)]

        CO2_info = CO2_info.append(CO2_pd)

    PM25_info = pd.DataFrame()
    for PM25_file_path in PM25_files_list:
        PM25_pd = pd.read_csv(PM25_file_path)
        # drop the last three digits in date (UTC)
        PM25_pd["date (UTC)"] = PM25_pd["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        PM25_pd["date-format"] = PM25_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

        # drop the empty value
        PM25_pd = PM25_pd[["date-format", "value"]]
        PM25_pd.rename(columns={'value': "PM25"}, inplace=True)
        PM25_pd["date-format"] = PM25_pd["date-format"] - pd.Timedelta(hours=5)
        PM25_pd = PM25_pd.dropna()

        PM25_pd = PM25_pd[(PM25_pd['date-format'] >= starting_date_time) &
                          (PM25_pd['date-format'] <= ending_date_time)]

        PM25_info = PM25_info.append(PM25_pd)

    VOC_info = pd.DataFrame()
    for VOC_file_path in VOC_files_list:
        VOC_pd = pd.read_csv(VOC_file_path)
        # drop the last three digits in date (UTC)
        VOC_pd["date (UTC)"] = VOC_pd["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        VOC_pd["date-format"] = VOC_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # drop the empty value
        VOC_pd = VOC_pd[["date-format", "value"]]
        VOC_pd.rename(columns={'value': "VOC"}, inplace=True)
        VOC_pd["date-format"] = VOC_pd["date-format"] - pd.Timedelta(hours=5)
        VOC_pd = VOC_pd.dropna()

        VOC_pd = VOC_pd[(VOC_pd['date-format'] >= starting_date_time) &
                        (VOC_pd['date-format'] <= ending_date_time)]

        VOC_info = VOC_info.append(VOC_pd)

    ###############################################################################################
    # add CO2 column to the data
    # need to be sorted first by date-format
    CO2_info = CO2_info.sort_values('date-format')

    df_CO2 = pd.merge_asof(IndoorTempHumidity, CO2_info['date-format'],
                           left_on="date_time", right_on='date-format',
                           allow_exact_matches=True, direction='backward')

    df_CO2.rename(columns={"date-format": "CO2-date-format-backward"}, inplace=True)

    df_CO2 = pd.merge_asof(df_CO2, CO2_info['date-format'],
                           left_on="date_time", right_on='date-format',
                           allow_exact_matches=True, direction='forward')

    df_CO2.rename(columns={"date-format": "CO2-date-format-forward"}, inplace=True)
    # drop the NA rows
    df_CO2 = df_CO2.dropna()
    # shift one unit down
    df_CO2['CO2-date-format-forward'] = df_CO2['CO2-date-format-forward'].shift(1)
    # drop the NA rows
    df_CO2 = df_CO2.dropna()

    # calculate the average value over the past 5 minutes
    CO2_avg = []
    for i in range(len(df_CO2)):
        cur_row = df_CO2.iloc[i]
        forward_time = cur_row['CO2-date-format-forward']
        backward_time = cur_row['CO2-date-format-backward']

        diff_back_forward = (backward_time - forward_time).total_seconds()
        diff_back_now = (cur_row['date_time'] - backward_time).total_seconds()

        if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 10:
            CO2_subset = CO2_info[
                (forward_time <= CO2_info['date-format']) & (CO2_info['date-format'] <= backward_time)]
            avg_CO2 = CO2_subset['CO2'].mean()
            print(avg_CO2)
        else:
            avg_CO2 = np.NAN
        CO2_avg.append(avg_CO2)
    df_CO2['CO2_avg'] = CO2_avg

    # add PM25 column
    # need to be sorted first
    PM25_info = PM25_info.sort_values('date-format')

    df_PM25 = pd.merge_asof(IndoorTempHumidity, PM25_info['date-format'],
                            left_on="date_time", right_on='date-format',
                            allow_exact_matches=True, direction='backward')
    df_PM25.rename(columns={"date-format": "PM25-date-format-backward"}, inplace=True)

    df_PM25 = pd.merge_asof(df_PM25, PM25_info['date-format'],
                            left_on="date_time", right_on='date-format',
                            allow_exact_matches=True, direction='forward')
    df_PM25.rename(columns={"date-format": "PM25-date-format-forward"}, inplace=True)

    # drop the NA rows
    df_PM25 = df_PM25.dropna()
    # shift one unit down
    df_PM25['PM25-date-format-forward'] = df_PM25['PM25-date-format-forward'].shift(1)
    # drop the NA rows
    df_PM25 = df_PM25.dropna()

    # calculate the average value over the past 5 minutes
    PM25_avg = []
    for i in range(len(df_PM25)):
        cur_row = df_PM25.iloc[i]
        forward_time = cur_row['PM25-date-format-forward']
        backward_time = cur_row['PM25-date-format-backward']

        diff_back_forward = (backward_time - forward_time).total_seconds()
        diff_back_now = (cur_row['date_time'] - backward_time).total_seconds()
        if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 10:
            PM25_subset = PM25_info[
                (forward_time <= PM25_info['date-format']) & (PM25_info['date-format'] <= backward_time)]
            avg_PM25 = PM25_subset['PM25'].mean()
            print(avg_PM25)
        else:
            avg_PM25 = np.NAN
        PM25_avg.append(avg_PM25)
    df_PM25['PM25_avg'] = PM25_avg

    # add VOC colunm
    # need to be sorted first
    VOC_info = VOC_info.sort_values('date-format')
    df_VOC = pd.merge_asof(IndoorTempHumidity, VOC_info['date-format'],
                           left_on="date_time", right_on='date-format',
                           allow_exact_matches=True, direction='backward')
    df_VOC.rename(columns={"date-format": "VOC-date-format-backward"}, inplace=True)

    df_VOC = pd.merge_asof(df_VOC, VOC_info['date-format'],
                           left_on="date_time", right_on='date-format',
                           allow_exact_matches=True, direction='forward')

    df_VOC.rename(columns={"date-format": "VOC-date-format-forward"}, inplace=True)
    # drop the NA rows
    df_VOC = df_VOC.dropna()
    # shift one unit down
    df_VOC['VOC-date-format-forward'] = df_VOC['VOC-date-format-forward'].shift(1)
    # drop the NA rows
    df_VOC = df_VOC.dropna()

    # calculate the average value over the past 5 minutes
    VOC_avg = []
    for i in range(len(df_VOC)):
        cur_row = df_VOC.iloc[i]
        forward_time = cur_row['VOC-date-format-forward']
        backward_time = cur_row['VOC-date-format-backward']
        diff_back_forward = (backward_time - forward_time).total_seconds()
        diff_back_now = (cur_row['date_time'] - backward_time).total_seconds()

        if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 10:
            VOC_subset = VOC_info[
                (forward_time <= VOC_info['date-format']) & (VOC_info['date-format'] <= backward_time)]
            avg_VOC = VOC_subset['VOC'].mean()
            print(avg_VOC)
        else:
            avg_VOC = np.NAN
        VOC_avg.append(avg_VOC)
    df_VOC['VOC_avg'] = VOC_avg

    # aggregate the result
    df_final = df_CO2.merge(df_PM25[["date_time", "PM25-date-format-forward", "PM25-date-format-backward", "PM25_avg"]],
                            left_on="date_time", right_on="date_time")

    df_final = df_final.merge(df_VOC[["date_time", "VOC-date-format-forward", "VOC-date-format-backward", "VOC_avg"]],
                              left_on="date_time", right_on="date_time")

    # then calculating the IAQ score
    # calculate the correlation matrix
    df_final_subset = df_final[
        ['Thermostat Temperature (F)', 'Thermostat Humidity (%RH)', 'CO2_avg', 'PM25_avg', 'VOC_avg']]
    corr_matrix = df_final_subset.corr()

    return df_final, corr_matrix


df_final, corr_matrix = IAQIndexCalculation(
    r'C:\Users\wuyi1234\Desktop\DataLogger\indoor sensor\EcobeeData\Sotiri\Hovardas Ecobee; 2021-01-09-to-2021-02-09.xlsx',
    r'C:\Users\wuyi1234\Desktop\DataLogger\target unit\630094',
    "WuhanIAQ_CO2",
    "WuhanIAQ_PM25",
    "WuhanIAQ_VOC")

import seaborn as sns

corr_heatmap = sns.heatmap(corr_matrix, cmap="PiYG", vmin=-1, vmax=1)
unit = '630094'
figure = corr_heatmap.get_figure()
figure.savefig(r"C:\Users\wuyi1234\Desktop\{}.png".format(unit),
               bbox_inches="tight",
               dpi=400)
