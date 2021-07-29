# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:17:54 2021

@author: wuyi1234
"""
import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detrend(df):
    new_col = []
    new_col.extend([np.nan])
    for i in range(1, len(df)):
        detrended_val = df.iloc[i, 6] - df.iloc[i - 1, 6]
        new_col.append(detrended_val)
    df['detrended_value'] = new_col
    return


def path_to_files(path):
    files = os.listdir(path)
    files_path = []
    files_name = []
    for file in files:
        files_path.append(os.path.join(path, file))
        # parse file name
        files_name.append(file[0:file.find(')') + 1])
    return files_path, files_name


file_paths, file_names = path_to_files(r'C:/Users/wuyi1234/Desktop/DataLogger/aaaa')


def plot_and_save(file_paths, file_names, save_path, ma=False):
    for file_path, file_name in zip(file_paths, file_names):
        CurrFile = pd.read_csv(file_path)
        # drop the last three digits in date (UTC)
        CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        CurrFile["date-format"] = CurrFile["date (UTC)"].map(
            lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # sort timestamp in ascending order
        CurrFile = CurrFile.sort_values(by='date-format', ascending=True)
        # delete the NA rows
        CurrFile = CurrFile.dropna()

        if ma == True:
            moving_average(CurrFile, 720, 3720)
            # define longer x-axis and shorter y-axis
            plt.figure(figsize=(10, 2))
            ax = plt.gca()
            plt.plot(CurrFile["date-format"], CurrFile['ma'], 'm')

            hrlocator = mdates.HourLocator()
            daylocator = mdates.HourLocator(byhour=[0, 6, 12, 18, 24], interval=1)

            majorFmt = mdates.DateFormatter('%Y-%m-%d, %H:%M')
            minorFmt = mdates.DateFormatter('%H:%M')

            ax.xaxis.set_major_locator(daylocator)
            ax.xaxis.set_major_formatter(majorFmt)
            # rotate 90 degrees
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

            ax.xaxis.set_minor_locator(hrlocator)
            ax.xaxis.set_minor_formatter(minorFmt)
            # rotate 90 degrees
            plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)

            ax.set_ylabel('CO2(ppm)', fontsize=13)
            # save prior to show
            file_name = file_name + "MA.svg"
            plt.savefig(os.path.join(save_path, file_name), format='svg', dpi=1200, bbox_inches='tight')
            plt.show()


        else:
            # define longer x-axis and shorter y-axis
            plt.figure(figsize=(10, 2))
            ax = plt.gca()
            plt.plot(CurrFile["date-format"], CurrFile['value'])

            hrlocator = mdates.HourLocator()
            daylocator = mdates.HourLocator(byhour=[0, 6, 12, 18, 24], interval=1)

            majorFmt = mdates.DateFormatter('%Y-%m-%d, %H:%M')
            minorFmt = mdates.DateFormatter('%H:%M')

            ax.xaxis.set_major_locator(daylocator)
            ax.xaxis.set_major_formatter(majorFmt)
            # rotate 90 degrees
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

            ax.xaxis.set_minor_locator(hrlocator)
            ax.xaxis.set_minor_formatter(minorFmt)
            # rotate 90 degrees
            plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)

            ax.set_ylabel('CO2(ppm)', fontsize=13)
            # save prior to show
            file_name = file_name + ".svg"
            plt.savefig(os.path.join(save_path, file_name), format='svg', dpi=1200, bbox_inches='tight')
            plt.show()


# plot_and_save(file_paths,file_names,"C:/Users/wuyi1234/Desktop/New",True)


'''
#built-in method
bb=CurrFile.iloc[:,6].rolling(window=360).mean()
'''


# draw all the plot in the single plot(30min-moving average)
def accumlating_plot(file_paths, file_names, save_path, unit_name):
    lines = []
    i = 0
    for file_path, file_name in zip(file_paths, file_names):
        CurrFile = pd.read_csv(file_path)
        # drop the last three digits in date (UTC)
        CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        CurrFile["date-format"] = CurrFile["date (UTC)"].map(
            lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # sort timestamp in ascending order
        CurrFile = CurrFile.sort_values(by='date-format', ascending=True)
        # delete the NA rows
        CurrFile = CurrFile.dropna()

        # calculate the moving average
        moving_average(CurrFile, 360, 1920)

        # detrend first
        detrend(CurrFile)
        # drop NA because of the calculating detrended value
        CurrFile = CurrFile.dropna()

        # line segment
        # change year,month,day to the same
        mindate = min(CurrFile['date-format']).date()
        # maxdate=max(CurrFile['date-format']).date())
        CurrFile['redefined-timestamp'] = CurrFile['date-format'].map(lambda x: x.replace(year=2000, month=1, day=1)
        if x.date() == mindate else x.replace(year=2000, month=1, day=2))
        lines.append(CurrFile[['date-format', 'redefined-timestamp', 'ma', 'detrended_value']])
        i += 1
        print(i)

    # then draw the line
    plt.figure(figsize=(10, 2))
    ax = plt.gca()
    for line in lines:
        plt.plot(line["redefined-timestamp"], line['detrended_value'], 'm')

    hrlocator = mdates.HourLocator()
    majorFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(hrlocator)
    ax.xaxis.set_major_formatter(majorFmt)
    # rotate 90 degrees
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set_ylabel('CO2(ppm)', fontsize=13)
    # save prior to show
    file_name = unit_name + ".svg"
    plt.savefig(os.path.join(save_path, file_name), format='svg', dpi=1200, bbox_inches='tight')

    plt.show()

    return


accumlating_plot(file_paths, file_names, "C:/Users/wuyi1234/Desktop/New", '61a3fa')
