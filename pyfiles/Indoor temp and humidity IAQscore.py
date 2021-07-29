# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:07:02 2021

@author: WuYi1234
"""


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
            if file == file_name:
                files_list.append((os.path.join(root, file)))
    return files_list


def CO2_index_calculation(CO2_Value):
    """
    Parameters
    ----------
    CO2_Value : measured CO2 value in ppm 
        
    Returns
    -------
    The AQI index in regard with CO2

    """
    AQI = 0
    # truncate the CO2_Value first
    CO2_Value = round(CO2_Value)
    if 0 <= CO2_Value <= 550:
        AQI = 100 - ((CO2_Value - 0) / (550 - 0)) * (100 - 75)
    if 550 <= CO2_Value <= 800:
        AQI = 75 - ((CO2_Value - 550) / (800 - 550)) * (75 - 50)
    if 800 <= CO2_Value <= 1350:
        AQI = 50 - ((CO2_Value - 800) / (1350 - 800)) * (50 - 25)
    if 1350 <= CO2_Value <= 1600:
        AQI = 25 - ((CO2_Value - 1350) / (1600 - 1350)) * (25 - 0)
    return round(AQI)


def PM25_index_calculation(PM25_value):
    """
    Parameters
    ----------
    PM10_value : measured CO2 value

    Returns
    -------
    The AQI index in regard with CO2
    """
    AQI = 0
    # truncate the PM25_value
    PM25_value = round(PM25_value, 1)
    if 0 <= PM25_value <= 12:
        AQI = 100 - ((PM25_value - 0) / (12 - 0)) * (100 - 75)
    if 12 <= PM25_value <= 35.5:
        AQI = 75 - ((PM25_value - 12) / (35.5 - 12)) * (75 - 50)
    if 35.5 <= PM25_value <= 55.5:
        AQI = 50 - ((PM25_value - 35.5) / (55.5 - 35.5)) * (50 - 25)
    if 55.5 <= PM25_value <= 67.5:
        AQI = 25 - ((PM25_value - 55.5) / (67.5 - 55.5)) * (25 - 0)
    return round(AQI)


def VOC_index_calculation(VOC_value):
    """
    Parameters
    ----------
    VOC_value : measured VOC in levels
    here VOC only has 0,1,2,3
    Returns
    -------
    The AQI index in regard with VOC

    """
    VOC_value = round(VOC_value)
    AQI = 0
    if VOC_value == 0:
        AQI = 100
    if VOC_value == 1:
        AQI = 75
    if VOC_value == 2:
        AQI = 50
    if VOC_value == 3:
        AQI = 25
    return round(AQI)


def RH_index_calculation(RH_value):
    """
    Parameters
    ----------
    RH_value : measured RH value

    Returns
    -------
    The AQI index in regard with RH
    """
    AQI = 0
    RH_value = round(RH_value)
    # first half
    if 0 <= RH_value <= 20:
        AQI = 0 + ((RH_value - 0) / (20 - 0)) * (25 - 0)
    if 20 <= RH_value <= 25:
        AQI = 25 + ((RH_value - 20) / (25 - 20)) * (50 - 25)
    if 25 <= RH_value <= 30:
        AQI = 50 + ((RH_value - 25) / (30 - 25)) * (75 - 50)
    if 30 <= RH_value <= 40:
        AQI = 75 + ((RH_value - 30) / (40 - 30)) * (100 - 75)

    # second half
    if 40 <= RH_value <= 50:
        AQI = 100 - ((RH_value - 40) / (50 - 40)) * (100 - 75)
    if 50 <= RH_value <= 60:
        AQI = 75 - ((RH_value - 50) / (60 - 50)) * (75 - 50)
    if 60 <= RH_value <= 70:
        AQI = 50 - ((RH_value - 60) / (70 - 60)) * (50 - 25)
    if 70 <= RH_value <= 100:
        AQI = 25 - ((RH_value - 70) / (100 - 70)) * (25 - 0)
    return round(AQI)


def temperature_index_calculation(temperature_value):
    """
    Parameters
    ----------
    temperature_value: measured temperature value (unit Fahrenheit)

    Returns
    -------
    The AQI index in regard with temperature
    """

    AQI = 0
    temperature_value = round(temperature_value, 1)
    # first half
    if 57.2 <= temperature_value <= 60.8:
        AQI = 0 + ((temperature_value - 57.2) / (60.8 - 57.2)) * (25 - 0)
    if 60.8 <= temperature_value <= 64.4:
        AQI = 25 + ((temperature_value - 60.8) / (64.4 - 60.8)) * (50 - 25)
    if 64.4 <= temperature_value <= 68.0:
        AQI = 50 + ((temperature_value - 64.4) / (68.0 - 64.4)) * (75 - 50)
    if 68.0 <= temperature_value <= 73.4:
        AQI = 75 + ((temperature_value - 68.0) / (73.4 - 68.0)) * (100 - 75)

    # second half
    if 73.4 <= temperature_value <= 78.8:
        AQI = 100 - ((temperature_value - 73.4) / (78.8 - 73.4)) * (100 - 75)
    if 78.8 <= temperature_value <= 80.6:
        AQI = 75 - ((temperature_value - 78.8) / (80.6 - 78.8)) * (75 - 50)
    if 80.6 <= temperature_value <= 82.4:
        AQI = 50 - ((temperature_value - 80.6) / (82.4 - 80.6)) * (50 - 25)
    if 82.4 <= temperature_value <= 84.2:
        AQI = 25 - ((temperature_value - 82.4) / (84.2 - 82.4)) * (25 - 0)
    return round(AQI)


# combine date and time
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


def Calculating_IAQ_index(folder_path,
                          CO2_file_name,
                          PM25_file_name,
                          VOC_file_name,
                          IndoorTempHumidity_file_path):
    """
    Parameters
    ----------
    folder_path : the same as the function file_searching
    file_name : the same as the function file_searching

    Returns
    -------
    results : the IAQ index

    """
    # read the data source
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
    # adding prior and after "padding" (extend the original )
    starting_date_time = starting_date_time - datetime.timedelta(minutes=1)
    ending_date_time = ending_date_time + datetime.timedelta(minutes=1)

    # find the file collections (file paths)
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
        CO2_pd = CO2_pd.dropna()
        CO2_pd = CO2_pd[["date-format", "value"]]
        CO2_pd.rename(columns={'value': "CO2"}, inplace=True)

        CO2_pd = CO2_pd[(CO2_pd['date-format'] >= starting_date_time) &
                        (CO2_pd['date-format'] <= ending_date_time)]

        CO2_info = CO2_info.append(CO2_pd)

    PM25_info = pd.DataFrame()
    for PM25_file_path in PM25_files_list:
        print(PM25_file_path)
        PM25_pd = pd.read_csv(PM25_file_path)
        # drop the last three digits in date (UTC)
        PM25_pd["date (UTC)"] = PM25_pd["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        PM25_pd["date-format"] = PM25_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # drop the empty value
        PM25_pd = PM25_pd.dropna()
        PM25_pd = PM25_pd[["date-format", "value"]]
        PM25_pd.rename(columns={'value': "PM25"}, inplace=True)

        PM25_pd = PM25_pd[(PM25_pd['date-format'] >= starting_date_time) &
                          (PM25_pd['date-format'] <= ending_date_time)]

        PM25_info = PM25_info.append(PM25_pd)

    VOC_info = pd.DataFrame()
    for VOC_file_path in VOC_files_list:
        print(VOC_file_path)
        VOC_pd = pd.read_csv(VOC_file_path)
        # drop the last three digits in date (UTC)
        VOC_pd["date (UTC)"] = VOC_pd["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        VOC_pd["date-format"] = VOC_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # drop the empty value
        VOC_pd = VOC_pd.dropna()
        VOC_pd = VOC_pd[["date-format", "value"]]
        VOC_pd.rename(columns={'value': "VOC"}, inplace=True)

        VOC_pd = VOC_pd[(VOC_pd['date-format'] >= starting_date_time) &
                        (VOC_pd['date-format'] <= ending_date_time)]

        VOC_info = VOC_info.append(VOC_pd)

    # add CO2 column
    # need to be sorted first
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
        if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 30:
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

        if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 30:
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

        if 270 <= diff_back_forward <= 330 and 0 <= diff_back_now <= 30:
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
    # calculate the IAQ index
    df_final = df_final.dropna()
    df_final['CO2_index'] = df_final['CO2_avg'].map(lambda x: CO2_index_calculation(x))
    df_final['PM25_index'] = df_final['PM25_avg'].map(lambda x: PM25_index_calculation(x))
    df_final['VOC_index'] = df_final['VOC_avg'].map(lambda x: VOC_index_calculation(x))
    df_final['humidity_index'] = df_final['Thermostat Humidity (%RH)'].map(lambda x: RH_index_calculation(x))
    df_final['temperature_index'] = df_final['Thermostat Temperature (F)'].map(
        lambda x: temperature_index_calculation(x))
    df_final['overall_index'] = df_final.apply(lambda row: row.CO2_index * 0.2
                                                           + row.PM25_index * 0.2
                                                           + row.VOC_index * 0.2
                                                           + row.humidity_index * 0.2
                                                           + row.temperature_index * 0.2, axis=1)

    return df_final


final_results = Calculating_IAQ_index("C:/Users/wuyi1234/Desktop/DataLogger/target unit/630091",
                                      "WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv",
                                      "WuhanIAQ_PM25_PM2.5_CUBIC_IAQ_PM25_part2.csv",
                                      "WuhanIAQ_VOC_VOC_CUBIC_IAQ_VOC_part2.csv",
                                      "C:/Users/wuyi1234/Desktop/DataLogger/indoor sensor/EcobeeData/DavidRussell/Russell Ecobee ereport-521750808280-2021-01-17-to-2021-02-17.xlsx")

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter, HoverTool, ColumnDataSource
from bokeh.layouts import column
from datetime import datetime
import pandas as pd
import os
from math import pi
import numpy as np

CDS_results = ColumnDataSource(data=final_results)

# visualization
unit_name = '6300c5'

time_interval = '5min-basis'
output_file("line.html")
from bokeh.models import Legend

p = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

CO2_line = p.line(x='date_time', y='CO2_index',
                  source=CDS_results, line_width=2, color="#E63721")

# add legend
legend = Legend(items=[
    ("CO2", [CO2_line])
], location="center")
# add hover tips
CO2_hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                ("y", "@CO2_index")],
                      formatters={'@date_time': 'datetime'})

p.add_tools(CO2_hover)
p.add_layout(legend, 'right')

p.xaxis.formatter = DatetimeTickFormatter(
    seconds=["%Y-%m-%d %H:%M:%S"],
    hours=["%Y-%m-%d %H:%M:%S"],
    days=["%Y-%m-%d %H:%M:%S"],
    months=["%Y-%m-%d %H:%M:%S"],
    years=["%Y-%m-%d %H:%M:%S"],
)
p.xaxis.major_label_orientation = pi / 4
p.xgrid.grid_line_color = None

"""
"""
p1 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p1.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

humidity_line = p1.line(x='date_time', y='humidity_index',
                        source=CDS_results, line_width=2, color="#1F0E95")

# add legend
legend = Legend(items=[
    ("humidity_line", [humidity_line])
    # ("temperature_line" , [temperature_line]),
    #  ("overall_line" , [overall_line])
], location="center")
# add hover tips
humidity_hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                     ("y", "@humidity_index")],
                           formatters={'@date_time': 'datetime'})

p1.add_tools(humidity_hover)
p1.add_layout(legend, 'right')

p1.xaxis.formatter = DatetimeTickFormatter(
    seconds=["%Y-%m-%d %H:%M:%S"],
    hours=["%Y-%m-%d %H:%M:%S"],
    days=["%Y-%m-%d %H:%M:%S"],
    months=["%Y-%m-%d %H:%M:%S"],
    years=["%Y-%m-%d %H:%M:%S"],
)
p1.xaxis.major_label_orientation = pi / 4
p1.xgrid.grid_line_color = None

"""
"""
p2 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p2.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

temperature_line = p2.line(x='date_time', y='temperature_index',
                           source=CDS_results, line_width=2, color="#F39E71")

# add legend
legend = Legend(items=[
    ("temperature_line", [temperature_line])
], location="center")

# add hover tips
temperature_hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                        ("y", "@temperature_index")],
                              formatters={'@date_time': 'datetime'})

p2.add_tools(temperature_hover)
p2.add_layout(legend, 'right')

p2.xaxis.formatter = DatetimeTickFormatter(
    seconds=["%Y-%m-%d %H:%M:%S"],
    hours=["%Y-%m-%d %H:%M:%S"],
    days=["%Y-%m-%d %H:%M:%S"],
    months=["%Y-%m-%d %H:%M:%S"],
    years=["%Y-%m-%d %H:%M:%S"],
)
p2.xaxis.major_label_orientation = pi / 4
p2.xgrid.grid_line_color = None

"""
"""
p3 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p3.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

VOC_line = p3.line(x='date_time', y='VOC_index',
                   source=CDS_results, line_width=2, color="#7828FD")

# add legend
legend = Legend(items=[
    ("VOC_line", [VOC_line])
], location="center")

# add hover tips
VOC_hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                ("y", "@VOC_index")],
                      formatters={'@date_time': 'datetime'})

p3.add_tools(VOC_hover)
p3.add_layout(legend, 'right')

p3.xaxis.formatter = DatetimeTickFormatter(
    seconds=["%Y-%m-%d %H:%M:%S"],
    hours=["%Y-%m-%d %H:%M:%S"],
    days=["%Y-%m-%d %H:%M:%S"],
    months=["%Y-%m-%d %H:%M:%S"],
    years=["%Y-%m-%d %H:%M:%S"],
)
p3.xaxis.major_label_orientation = pi / 4
p3.xgrid.grid_line_color = None

"""
"""

p4 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p4.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

PM25_line = p4.line(x='date_time', y='PM25_index',
                    source=CDS_results, line_width=2, color="#337704")
# add legend
legend = Legend(items=[
    ("PM25", [PM25_line])
], location="center")
# add hover tips
PM25_hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                 ("y", "@PM25_index")],
                       formatters={'@date_time': 'datetime'})

p4.add_tools(PM25_hover)
p4.add_layout(legend, 'right')

p4.xaxis.formatter = DatetimeTickFormatter(
    seconds=["%Y-%m-%d %H:%M:%S"],
    hours=["%Y-%m-%d %H:%M:%S"],
    days=["%Y-%m-%d %H:%M:%S"],
    months=["%Y-%m-%d %H:%M:%S"],
    years=["%Y-%m-%d %H:%M:%S"],
)
p4.xaxis.major_label_orientation = pi / 4
p4.xgrid.grid_line_color = None

"""
"""
p5 = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p5.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

overall_line = p5.line(x='date_time', y='overall_index',
                       source=CDS_results, line_width=2, color="#5BDEB4")

# add legend
legend = Legend(items=[
    ("overall_line", [overall_line])
], location="center")

# add hover tips
overall_hover = HoverTool(tooltips=[("x", "@date_time{%Y-%m-%d %H:%M:%S}"),
                                    ("y", "@overall_index")],
                          formatters={'@date_time': 'datetime'})

p5.add_tools(overall_hover)
p5.add_layout(legend, 'right')

p5.xaxis.formatter = DatetimeTickFormatter(
    seconds=["%Y-%m-%d %H:%M:%S"],
    hours=["%Y-%m-%d %H:%M:%S"],
    days=["%Y-%m-%d %H:%M:%S"],
    months=["%Y-%m-%d %H:%M:%S"],
    years=["%Y-%m-%d %H:%M:%S"],
)
p5.xaxis.major_label_orientation = pi / 4
p5.xgrid.grid_line_color = None

# put all the plots in a VBox
final_plot = column(p, p1, p2, p3, p4, p5)
show(final_plot)
