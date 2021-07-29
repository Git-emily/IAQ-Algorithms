# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:04:07 2021

@author: WuYi
"""

import os
from datetime import datetime
from math import pi

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, HoverTool, ColumnDataSource
from bokeh.plotting import figure


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


"""
Note: for all index component score, the higher the better.
100 is the upper bound.
"""


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


def split_dataset(data_set, batch_size):
    num_batch = len(data_set) // batch_size
    batches = []
    for i in range(num_batch):
        batches.append(data_set[batch_size * i:batch_size * (i + 1)])
    return batches


def EntropyWeight(df):
    # numerical stability because of the division
    df = df + 1
    columnSum = df.sum(axis=0)
    # print(columnSum)
    df = df / columnSum
    factor = -1 / np.log(len(df))
    Entropy = factor * ((df * np.log(df)).sum(axis=0))
    DiversityDegree = 1 - Entropy
    NormalizedWeight = DiversityDegree / DiversityDegree.sum()
    return NormalizedWeight


def Calculating_IAQ_index(folder_path, CO2_file_name, PM10_file_name, VOC_file_name, humidity_file_name,
                          temperature_file_name, batch_size):
    """
    Parameters
    ----------
    folder_path : the same as the function file_searching
    file_name : the same as the function file_searching

    Returns
    -------
    results : the IAQ index

    """

    # find the file collections
    CO2_files_list = file_searching(folder_path, CO2_file_name)
    PM10_files_list = file_searching(folder_path, PM10_file_name)
    VOC_files_list = file_searching(folder_path, VOC_file_name)
    humidity_files_list = file_searching(folder_path, humidity_file_name)
    temperature_files_list = file_searching(folder_path, temperature_file_name)
    results = []

    # deal day by day
    for CO2_file_path in CO2_files_list:
        path = os.path.normpath(CO2_file_path)
        path = path.split(os.sep)  ##os.sep ='\\'
        sampling_date = path[-3]

        CO2_pd = pd.read_csv(CO2_file_path)
        # drop the last three digits in date (UTC)
        CO2_pd["date (UTC)"] = CO2_pd["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        CO2_pd["date-format"] = CO2_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # drop the empty value
        CO2_pd = CO2_pd.dropna()
        CO2_pd = CO2_pd[["date-format", "value"]]
        CO2_pd.rename(columns={'value': "CO2"}, inplace=True)

        # find the rest corresponding files according to sampling_data e.g 'datalogger6300c5(2021.01.11-2021.01.11)'
        for PM10_file_path in PM10_files_list:
            if sampling_date in PM10_file_path:
                print(PM10_file_path)
                PM10_pd = pd.read_csv(PM10_file_path)
                # drop the last three digits in date (UTC)
                PM10_pd["date (UTC)"] = PM10_pd["date (UTC)"].map(lambda x: x[:-4])
                # convert your timestamps to datetime and then use matplotlib
                PM10_pd["date-format"] = PM10_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
                # drop the empty value
                PM10_pd = PM10_pd.dropna()
                PM10_pd = PM10_pd[["date-format", "value"]]
                PM10_pd.rename(columns={'value': "PM10"}, inplace=True)

        for VOC_file_path in VOC_files_list:
            if sampling_date in VOC_file_path:
                print(VOC_file_path)
                VOC_pd = pd.read_csv(VOC_file_path)
                # drop the last three digits in date (UTC)
                VOC_pd["date (UTC)"] = VOC_pd["date (UTC)"].map(lambda x: x[:-4])
                # convert your timestamps to datetime and then use matplotlib
                VOC_pd["date-format"] = VOC_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
                # drop the empty value
                VOC_pd = VOC_pd.dropna()
                VOC_pd = VOC_pd[["date-format", "value"]]
                VOC_pd.rename(columns={'value': "VOC"}, inplace=True)

        for humidity_file_path in humidity_files_list:
            if sampling_date in humidity_file_path:
                print(humidity_file_path)
                humidity_pd = pd.read_csv(humidity_file_path)
                # drop the last three digits in date (UTC)
                humidity_pd["date (UTC)"] = humidity_pd["date (UTC)"].map(lambda x: x[:-4])
                # convert your timestamps to datetime and then use matplotlib
                humidity_pd["date-format"] = humidity_pd["date (UTC)"].map(
                    lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
                # drop the empty value
                humidity_pd = humidity_pd.dropna()
                humidity_pd = humidity_pd[["date-format", "value"]]
                humidity_pd.rename(columns={'value': "humidity"}, inplace=True)

        for temperature_file_path in temperature_files_list:
            if sampling_date in temperature_file_path:
                print(temperature_file_path)
                temperature_pd = pd.read_csv(temperature_file_path)
                # drop the last three digits in date (UTC)
                temperature_pd["date (UTC)"] = temperature_pd["date (UTC)"].map(lambda x: x[:-4])
                # convert your timestamps to datetime and then use matplotlib
                temperature_pd["date-format"] = temperature_pd["date (UTC)"].map(
                    lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
                # drop the empty value
                temperature_pd = temperature_pd.dropna()
                temperature_pd = temperature_pd[["date-format", "value"]]
                temperature_pd.rename(columns={'value': "temperature"}, inplace=True)

        # starting to merge
        merged_pd = CO2_pd.merge(PM10_pd, how='inner', on='date-format')
        merged_pd = merged_pd.merge(VOC_pd, how='inner', on='date-format')
        merged_pd = merged_pd.merge(humidity_pd, how='inner', on='date-format')
        merged_pd = merged_pd.merge(temperature_pd, how='inner', on='date-format')

        # calculate the separete the IAQ index and the overall index
        # first split the dataset
        batches = split_dataset(merged_pd, batch_size)

        for batch in batches:
            # select min_time as the start time of the batch
            min_time = min(batch["date-format"])
            # using average value to represent the time interval
            avg_CO2 = batch["CO2"].mean()
            avg_PM10 = batch["PM10"].mean()
            avg_VOC = batch["VOC"].mean()
            avg_humidity = batch["humidity"].mean()
            avg_temperature = batch["temperature"].mean()
            # calculate the IAQ index
            CO2_index = CO2_index_calculation(avg_CO2)
            PM10_index = PM10_index_calculation(avg_PM10)
            VOC_index = VOC_index_calculation(avg_VOC)
            humidity_index = RH_index_calculation(avg_humidity)
            temperature_index = temperature_index_calculation(avg_temperature)
            # calculate the overall scores excluding the VOC
            # uniform distributed weighting schedule
            # overall_scores=0.2*CO2_index+0.2*PM10_index+0.2*humidity_index+0.2*temperature_index+0.2*VOC_index
            # The entropy value method
            # first create dataframe for the calculation excluding the "date-format" column
            EVM_df = batch[["CO2", "PM10", "VOC", "humidity", "temperature"]]
            weights = EntropyWeight(EVM_df)
            weights = pd.DataFrame(weights).transpose()
            overall_scores = weights["CO2"] * CO2_index
            +weights["PM10"] * PM10_index
            +weights["VOC"] * humidity_index
            +weights["humidity"] * temperature_index
            +weights["temperature"] * VOC_index
            # print(weights)
            # store the results
            results.append(pd.DataFrame({"timestamp": [min_time],
                                         "CO2_index": [CO2_index],
                                         "PM10_index": [PM10_index],
                                         "VOC_index": [VOC_index],
                                         "humidity_index": [humidity_index],
                                         "temperature_index": [temperature_index],
                                         "overall_scores": [overall_scores]}))
    return results


# select appropriate index calculator
# CO2_file_name="WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv"
# PM10_file_name="WuhanIAQ_PM10_PM10_CUBIC_IAQ_PM10_part2.csv"
# VOC_file_name="WuhanIAQ_VOC_VOC_CUBIC_IAQ_VOC_part2.csv"
# humidity_file_name="WuhanIAQ_RH_Humidity_CUBIC_IAQ_RH_part2.csv"
# temperature_file_name="WuhanIAQ_Temp_Temperature_CUBIC_IAQ_T_part2.csv"


CO2_file_name = "WuhanIAQ_CO2_CO2 concentration_WUHAN_IAQ_CO2_part2.csv"
PM10_file_name = "WuhanIAQ_PM10_PM10_WUHANIAQ_PM10_part2.csv"
VOC_file_name = "WuhanIAQ_VOC_VOC_WUHAN_IAQ_VOC_part2.csv"
# humidity_file_name="WuhanIAQ_RH_Humidity_WUHAN_IAQ_H_DK_part2.csv"
humidity_file_name = "WuhanIAQ_RH_Humidity_WUHANIAQ_HUM_part2.csv"
temperature_file_name = "WuhanIAQ_Temp_Temperature_WUHAN_IAQ_T_part2.csv"

# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c5"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c4"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/630198"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c8"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/63009b"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300cc"
# folder_path="C:/Users/wuyi1234/Desktop/DataLogger/target unit/63011d"
folder_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6309ec"

batch_size = 12  # 1 mins
batch_size = 720  # 1 hour

results = Calculating_IAQ_index(folder_path,
                                CO2_file_name,
                                PM10_file_name,
                                VOC_file_name,
                                humidity_file_name,
                                temperature_file_name,
                                batch_size)

# plot
# transform the results to dataFrame
index_results = pd.DataFrame(columns=['timestamp',
                                      'CO2_index',
                                      'PM10_index',
                                      'VOC_index',
                                      'humidity_index',
                                      'temperature_index',
                                      'overall_scores'])

for item in results:
    tempDf = pd.DataFrame({"timestamp": [item["timestamp"][0]],
                           "CO2_index": [item["CO2_index"][0]],
                           "PM10_index": [item["PM10_index"][0]],
                           "VOC_index": [item["VOC_index"][0]],
                           "humidity_index": [item["humidity_index"][0]],
                           "temperature_index": [item["temperature_index"][0]],
                           "overall_scores": [item["overall_scores"][0]]})

    index_results = index_results.append(tempDf, ignore_index=True)

CDS_results = ColumnDataSource(data=index_results)

# unit_name='6300c5'
# unit_name='6300c4'
# unit_name='630198'
# unit_name='6300c8'
# unit_name='63009b'
# unit_name='62ff02'
# unit_name='6300cc'
# unit_name='63011d'
unit_name = '6309ec'

time_interval = 'Hour-basis'
output_file("line.html")
from bokeh.models import Legend

p = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-5, 100))
p.yaxis.ticker = [-5, 0, 25, 50, 75, 100]

# bands in the middle
# good_box = BoxAnnotation(top=25,bottom=0, fill_alpha=0.7, fill_color='#00fff2')
# moderate_box = BoxAnnotation(top=50,bottom=25,fill_alpha=0.7, fill_color='#00ff00')
# unhealthy_box = BoxAnnotation(top=75,bottom=50, fill_alpha=0.7, fill_color='#ffff00')
# hazardous_box = BoxAnnotation(top=100,bottom=75, fill_alpha=0.7, fill_color='#ff0000')

# for VOC plot there would in some individual
# color background
# p.add_layout(good_box)
# p.add_layout(moderate_box)
# p.add_layout(unhealthy_box)
# p.add_layout(hazardous_box)

CO2_line = p.line(x='timestamp', y='CO2_index',
                  source=CDS_results, line_width=2, color="#E63721")

# add legend
legend = Legend(items=[
    ("CO2", [CO2_line])
], location="center")
# add hover tips
CO2_hover = HoverTool(tooltips=[("x", "@timestamp{%Y-%m-%d %H:%M:%S}"),
                                ("y", "@CO2_index")],
                      formatters={'@timestamp': 'datetime'})

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

humidity_line = p1.line(x='timestamp', y='humidity_index',
                        source=CDS_results, line_width=2, color="#1F0E95")

# add legend
legend = Legend(items=[
    ("humidity_line", [humidity_line])
    # ("temperature_line" , [temperature_line]),
    #  ("overall_line" , [overall_line])
], location="center")
# add hover tips
humidity_hover = HoverTool(tooltips=[("x", "@timestamp{%Y-%m-%d %H:%M:%S}"),
                                     ("y", "@humidity_index")],
                           formatters={'@timestamp': 'datetime'})

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

temperature_line = p2.line(x='timestamp', y='temperature_index',
                           source=CDS_results, line_width=2, color="#F39E71")

# add legend
legend = Legend(items=[
    ("temperature_line", [temperature_line])
], location="center")

# add hover tips
temperature_hover = HoverTool(tooltips=[("x", "@timestamp{%Y-%m-%d %H:%M:%S}"),
                                        ("y", "@temperature_index")],
                              formatters={'@timestamp': 'datetime'})

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

VOC_line = p3.line(x='timestamp', y='VOC_index',
                   source=CDS_results, line_width=2, color="#7828FD")

# add legend
legend = Legend(items=[
    ("VOC_line", [VOC_line])
], location="center")

# add hover tips
VOC_hover = HoverTool(tooltips=[("x", "@timestamp{%Y-%m-%d %H:%M:%S}"),
                                ("y", "@VOC_index")],
                      formatters={'@timestamp': 'datetime'})

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

PM10_line = p4.line(x='timestamp', y='PM10_index',
                    source=CDS_results, line_width=2, color="#337704")
# add legend
legend = Legend(items=[
    ("PM10", [PM10_line])
], location="center")
# add hover tips
PM10_hover = HoverTool(tooltips=[("x", "@timestamp{%Y-%m-%d %H:%M:%S}"),
                                 ("y", "@PM10_index")],
                       formatters={'@timestamp': 'datetime'})

p4.add_tools(PM10_hover)
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

overall_line = p5.line(x='timestamp', y='overall_scores',
                       source=CDS_results, line_width=2, color="#5BDEB4")

# add legend
legend = Legend(items=[
    ("overall_line", [overall_line])
], location="center")

# add hover tips
overall_hover = HoverTool(tooltips=[("x", "@timestamp{%Y-%m-%d %H:%M:%S}"),
                                    ("y", "@overall_scores")],
                          formatters={'@timestamp': 'datetime'})

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
