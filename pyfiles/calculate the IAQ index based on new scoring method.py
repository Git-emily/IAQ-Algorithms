# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:31:35 2021

@author: wuyi1234
"""

import datetime
import os
import warnings
from math import pi

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
# from datetime import datetime
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, Legend, HoverTool, ColumnDataSource
from bokeh.plotting import figure
from pythermalcomfort.models import pmv

PATH = os.path.abspath(os.path.dirname(os.getcwd()))
warnings.filterwarnings("ignore", category=DeprecationWarning)


def Fahrenheit2Celsius(Fahrenheit):
    return (Fahrenheit - 32) * (5 / 9)


def ThermalComfort(temperature, humidity):
    """
    temperature value in Fahrenheit
    relative humidity value in % 
    
    Returns
    -------
    thermal comfort
    """
    celsius = Fahrenheit2Celsius(temperature)
    pmv_value = pmv(tdb=celsius, tr=celsius, vr=0.2, rh=humidity, met=1.1, clo=0.6,
                    standard='ISO', units='SI')
    if abs(pmv_value) <= 1:
        pmv_score = abs(pmv_value) * (-30) + 105
        pmv_score = np.min([pmv_score, 100])
    if abs(pmv_value) > 1:
        pmv_score = abs(pmv_value) * (-15) + 90
        pmv_score = np.max([pmv_score, 0])
    return pmv_score


def Ventilation(CO2):
    """
    CO2 value in ppm
    
    Returns
    -------
    ventilation score
    """
    if CO2 <= 900:
        ventilation_score = CO2 * (-0.1) + 165
        ventilation_score = np.min([ventilation_score, 100])
    if CO2 > 900:
        ventilation_score = CO2 * (-0.15) + 210
        ventilation_score = np.max([ventilation_score, 0])
    return ventilation_score


def AirQuality(PM25, VOC):
    """
    PM2.5 in ug/m3
    TVOC/VOC  in ug/m3
    """
    # calculate PM2.5 first
    if PM25 <= 15:
        PM25_score = PM25 * (-5) + 150
        PM25_score = np.min([PM25_score, 100])
    if PM25 > 15:
        PM25_score = PM25 * (-0.75) + 86.25
        PM25_score = np.max([PM25_score, 0])

    # then calculate VOC
    VOC = round(VOC)
    if VOC == 0:
        # it corresponds to 150 ug/m3
        VOC_score = 150 * (-0.05) + 100
        VOC_score = np.min([VOC_score, 100])
    if VOC == 1:
        VOC_score = 650 * (-0.01) + 80
        VOC_score = np.max([VOC_score, 0])
    if VOC == 2:
        VOC_score = 2000 * (-0.01) + 80
        VOC_score = np.max([VOC_score, 0])
    if VOC == 3:
        VOC_score = 6500 * (-0.01) + 80
        VOC_score = np.max([VOC_score, 0])

    return np.min([PM25_score, VOC_score])


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


def Data_PreProcess(IndoorTempHumidity_file_path,folder_path,files_list):  # which contains the CO2,PM2.5 and VOC files

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
    item_info = pd.DataFrame()
    items = []
    for file_name in files_list:
        All_files = file_searching(folder_path,files_list[file_name])
        item = file_name.split('_')[0]
        items.append(item)
        for file_path in All_files:
            # item_pd = item+'_pd'
            item_pd = pd.read_csv(file_path)
            # drop the last three digits in date (UTC)
            item_pd["date (UTC)"] = item_pd["date (UTC)"].map(lambda x: x[:-4])
            # convert your timestamps to datetime and then use matplotlib
            item_pd["date-format"] = item_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

            # drop the empty value
            item_pd = item_pd[["date-format", "value"]]
            item_pd.rename(columns={'value': item}, inplace=True)
            item_pd["date-format"] = item_pd["date-format"] - pd.Timedelta(hours=5)  # CDT = GMT-5, EDT=GMT-4
            item_pd = item_pd.dropna()

            item_pd = item_pd[(item_pd['date-format'] >= starting_date_time) &
                            (item_pd['date-format'] <= ending_date_time)]

            item_info = item_info.append(item_pd)
    # need to be sorted first by date-format
    item_info = item_info.sort_values('date-format')

    df_item = pd.merge_asof(IndoorTempHumidity, item_info['date-format'],
                           left_on="date_time", right_on='date-format',
                           allow_exact_matches=True, direction='backward')

    df_item.rename(columns={"date-format": "date-format-backward"}, inplace=True)

    df_item = pd.merge_asof(df_item, item_info['date-format'],
                           left_on="date_time", right_on='date-format',
                           allow_exact_matches=True, direction='forward')

    df_item.rename(columns={"date-format": "date-format-forward"}, inplace=True)
    # drop the NA rows
    # shift one unit down
    df_item['date-format-forward'] = df_item['date-format-forward'].shift(1)
    # drop the NA rows
    df_item = df_item.dropna()

    # calculate the average value over the past 5 minutes
    avg = {}
    for i in items:
        avg[i] = []
    for i in range(len(df_item)):
        cur_row = df_item.iloc[i]
        forward_time = cur_row['date-format-forward']
        backward_time = cur_row['date-format-backward']

        diff_back_forward = (backward_time - forward_time).total_seconds()
        diff_back_now = (cur_row['date_time'] - backward_time).total_seconds()

        for avg_name in avg:
            if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 10:
                subset= item_info[(forward_time <= item_info['date-format']) & (item_info['date-format'] <= backward_time)]
                avg_value = subset[avg_name].mean()
                print(avg_value)
                avg[avg_name].append(avg_value)
            else:
                avg[avg_name].append(np.NAN)
    for avg_list in avg:
        df_item[avg_list+'_avg'] = avg[avg_list]

    df_final = df_item.dropna()
    return df_final

# def Redefined_Time(date_time):
#     if

def IAQIndexCalculation(df_final,starting_hour):
    # then calculating the IAQ score
    # calculate the IAQ index
    df_final['thermal_comfort'] = df_final.apply(lambda row: ThermalComfort(row['Thermostat Temperature (F)'],
                                                                            row['Thermostat Humidity (%RH)']), axis=1)
    df_final['ventilation'] = df_final['CO2_avg'].map(lambda x: Ventilation(x))
    df_final['air_quality'] = df_final.apply(lambda row: AirQuality(row['PM25_avg'],
                                                                    row['VOC_avg']), axis=1)

    df_final['overall_index'] = df_final.apply(lambda row: row['thermal_comfort'] * (1 / 3)
                                                           + row['ventilation'] * (1 / 3)
                                                           + row['air_quality'] * (1 / 3), axis=1)
    df_final['Week_Day'] = df_final['date_time'].map(
        lambda x: pd.Timestamp(x.year, x.month, x.day, 0, 0, 0).weekday())

    # df_final['overall_index'] = df_final.apply(lambda row: row['ventilation'] * (1 / 2)
    #                                                        + row['air_quality'] * (1 / 2), axis=1)

    # df_final['overall_index'] = df_final.apply(lambda row: row['thermal_comfort'] * (1 / 2)
    #                                                        + row['ventilation'] * (1 / 2), axis=1)

    # df_final['overall_index'] = df_final.apply(lambda row: row['thermal_comfort'] * (1 / 2)
    #                                                        + row['air_quality'] * (1 / 2), axis=1)


    # divide the df_final into daily curves
    min_time = min(df_final['date_time'])
    max_time = max(df_final['date_time'])
    # each element in file_list would be one day-data
    file_list = []
    # each element in date_list would be corresponding to weekday of the 24-hour
    # len(date_list) should be equal to the file_list
    date_list = []
    time_list = []

    # the starter
    starting_date = pd.Timestamp(min_time.year, min_time.month, min_time.day,
                                 starting_hour, 0, 0)

    ending_date = starting_date + pd.Timedelta(days=1)

    while ending_date < max_time + pd.Timedelta(days=1):
        filtered_file = df_final[(starting_date <= df_final["date_time"]) &
                                 (df_final["date_time"] < ending_date)]
        if len(filtered_file) != 0:
            filtered_file = filtered_file.sort_values(by='date_time', ascending=True)
            # remove duplicate rows if there is
            filtered_file = filtered_file.drop_duplicates()
            file_list.append(filtered_file)
            if starting_date.weekday() in [0, 1, 2, 3, 4]:
                date_list.append('weekday')
            else:
                date_list.append('weekend')

            time_list.append([starting_date, ending_date])

        starting_date += pd.Timedelta(days=1)
        ending_date += pd.Timedelta(days=1)

        print(starting_date)
        print(ending_date)

    print('start return')
    try:
        return df_final, file_list, date_list, time_list
    except Exception as e:
        print(e)

def Overall_visualization(df_final, file_list, date_list, time_list, unit_name, time_interval):
    """
    visualization
    Overall
    """
    CDS_results = ColumnDataSource(data=df_final)  # Bokeh data format

    # p is Thermal comfort, p1 is ventilation, p2 is air_quality,p3 is overall_index
    p = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
    p1 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
    p2 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
    p3 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
    Ps = [p,p1,p2,p3]
    plots_names = ['thermal_comfort', 'ventilation', 'air_quality', 'overall_index']
    color_palette = ['#E63721', '#21E6CF', '#60EB10', '#EB10BC']
    for i,plo in enumerate(Ps):
        plo.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

        line = plo.line(x='date_time', y= plots_names[i],
                                      source=CDS_results, line_width=2, color=color_palette[i])
        # add legend
        legend = Legend(items=[(plots_names[i], [line])], location="center")
        # add hover tips
        hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                                    ("y", "@"+plots_names[i])],
                                          formatters={'@date_time': 'datetime'})
        plo.add_tools(hover)
        plo.add_layout(legend, 'right')

        plo.xaxis.formatter = DatetimeTickFormatter(
            seconds=["%Y-%m-%d %H:%M:%S"],
            hours=["%Y-%m-%d %H:%M:%S"],
            days=["%Y-%m-%d %H:%M:%S"],
            months=["%Y-%m-%d %H:%M:%S"],
            years=["%Y-%m-%d %H:%M:%S"],
        )
        plo.xaxis.major_label_orientation = pi / 4
        plo.xgrid.grid_line_color = None

    # put all the plots in a VBox
    final_plot = column(p, p1, p2, p3)
    output_file("line.html")
    show(final_plot)

    """
    weekday and weekend pattern
    """
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
        # change year,month,day to the same
        mindate = time_list[index][0].date()
        file_list[index]['redefined-timestamp'] = file_list[index]['date_time'].map(
            lambda x: x.replace(year=2000)
            if x.date() == mindate else x.replace(year=2000))
        weekday_lines.append(file_list[index])

    for index in weekend_index:
        mindate = time_list[index][0].date()
        file_list[index]['redefined-timestamp'] = file_list[index]['date_time'].map(
            lambda x: x.replace(year=2000)
            if x.date() == mindate else x.replace(year=2000))
        weekend_lines.append(file_list[index])

    if not os.path.exists(PATH + r'\export_IAQCalcu'):
        os.mkdir(PATH + r'\export_IAQCalcu')

    for col, color in zip(plots_names, color_palette):
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 9))
        fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(19, 9))
        for line in weekday_lines:
            ax1.plot(line["redefined-timestamp"], line[col], color)
            # ax1.yaxis.set_ticks([0,10,20,30,40,50,60,70,80,90,100])

        for line in weekend_lines:
            ax2.plot(line["redefined-timestamp"], line[col], color)
            # ax2.yaxis.set_ticks([0,10,20,30,40,50,60,70,80,90,100])

        #Add by emily
        lines = weekday_lines + weekend_lines
        for line in lines:
            ax3.plot(line["redefined-timestamp"], line[col], color)


        hrlocator = mdates.HourLocator()  # date interval
        majorFmt = mdates.DateFormatter('%H:%M')  # set the formatter of X_axis
        ax1.xaxis.set_major_locator(hrlocator)
        ax1.xaxis.set_major_formatter(majorFmt)
        ax2.xaxis.set_major_locator(hrlocator)
        ax2.xaxis.set_major_formatter(majorFmt)

        ax3.xaxis.set_major_locator(hrlocator)
        ax3.xaxis.set_major_formatter(majorFmt)
        # rotate 90 degrees
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        ax1.set_ylabel('weekday_{}'.format(col), fontsize=13)
        ax2.set_ylabel('weekend_{}'.format(col), fontsize=13)

        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90)
        ax3.set_ylabel('week_{}'.format(col), fontsize=13)

        # save prior to show
        file_name = unit_name + col + ".svg"
        plt.savefig(os.path.join(PATH + r'\export_IAQCalcu', file_name), format='svg', dpi=1200, bbox_inches='tight')

        plt.show()

    """
    histogram
    """
    # overall
    for col, color in zip(plots_names, color_palette):
        plt.hist(df_final[col], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 color=color)  # X_axis is data, Y_axis is the times of data
        plt.xticks(range(0, 110, 10))
        plt.title(unit_name + '_' + col)
        file_name = unit_name + col + "_overall_hist.svg"
        plt.savefig(os.path.join(PATH + r'\export_IAQCalcu', file_name), format='svg', dpi=1200, bbox_inches='tight')
        plt.show()

    # weekday & weekend
    weekday_accumulation = pd.DataFrame()
    weekend_accumulation = pd.DataFrame()
    for line in weekday_lines:
        weekday_accumulation = weekday_accumulation.append(line)

    for line in weekend_lines:
        weekend_accumulation = weekend_accumulation.append(line)

    for col, color in zip(plots_names, color_palette):
        fig, (ax11, ax22) = plt.subplots(2, 1)

        ax11.hist(weekday_accumulation[col], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], color=color)
        ax22.hist(weekend_accumulation[col], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], color=color)

        ax11.set_xticks(range(0, 110, 10))
        ax22.set_xticks(range(0, 110, 10))

        ax11.set_ylabel('weekday', fontsize=13)
        ax22.set_ylabel('weekend', fontsize=13)

        ax22.set_xlabel(col, fontsize=13)

        # save prior to show
        file_name = unit_name + col + "_hist.svg"
        plt.savefig(os.path.join(PATH + r'\export_IAQCalcu', file_name), format='svg', dpi=1200, bbox_inches='tight')

        plt.show()


if __name__ == '__main__':
    EcobeeData_name = '\\indoor sensor\\EcobeeData\\Sotiri\\Hovardas Ecobee; 2021-01-09-to-2021-02-09.xlsx'
    unit_name = '630094'
    files_list = {'CO2_files_list': 'WuhanIAQ_CO2','PM25_file_list' : 'WuhanIAQ_PM25','VOC_file_list' : 'WuhanIAQ_VOC'}
    time_interval = '5min-basis'
    curve_starting_hour = 11  # it is 24-hour format

    df_final = Data_PreProcess(PATH  + EcobeeData_name,PATH + '\\target unit(this is demo data)\\'+unit_name, files_list)
    df_final, file_list, date_list, time_list = IAQIndexCalculation(df_final,curve_starting_hour)
    Overall_visualization(df_final, file_list, date_list, time_list, unit_name, time_interval)
