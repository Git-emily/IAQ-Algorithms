# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:45:01 2021

@author: WuYi
"""
from datetime import datetime
from math import pi

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import DatetimeTickFormatter, Legend, HoverTool, ColumnDataSource
from bokeh.plotting import figure
from scipy.stats import norm

# CO2_pd=pd.read_csv(
#    'C:/Users/wuyi1234/Desktop/DataLogger/target unit/61a52b/datalogger61a52b(2021.01.18-2021.01.19)/WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2/WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv')


CO2_pd = pd.read_csv(
    'C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c8/datalogger6300c8(2021.01.18-2021.01.19)/Thermocouple_T_MCP9600_Temperature_TC_0X60/Thermocouple_T_MCP9600_Temperature_TC_0X60_part2.csv')

# preprocess the timestamp in the anomaly list
# drop the last three digits in date (UTC)
CO2_pd["date (UTC)"] = CO2_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
CO2_pd["x"] = CO2_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))


def UnivariateGaussian(window_size, df, threshold):
    # initialize the window
    window = np.array([])
    anomaly_list = pd.DataFrame()

    for row_index in range(len(df)):
        current_row = df.iloc[row_index]

        if len(window) == window_size:
            mu = window.mean()
            sigma = window.std()

            if sigma == 0.0:
                sigma = 0.1

            # calculate the probability of new element
            prob = norm.pdf(current_row.value, mu, sigma)

            # prob=abs(np.log(prob))
            # print(abs(np.log(prob)))

            # print(prob)
            if prob != prob:
                print(prob)

            if prob < threshold:
                # flag it as anomaly
                anomaly_list = anomaly_list.append(current_row)

            else:
                # remove the old data and append the new data
                window = window[1:]
                window = np.append(window, current_row.value)

        else:
            # append the data
            window = np.append(window, current_row.value)

    return anomaly_list


# anomaly_list=UnivariateGaussian(6,CO2_pd,1e-10)


# def GMM():


def HLPassFilter(window_size, df, alpha):
    # initialize the window
    window = np.array([])
    anomaly_list = pd.DataFrame()

    for row_index in range(len(df)):
        current_row = df.iloc[row_index]

        if len(window) == window_size:
            avg = window.mean()
            std = window.std()

            if std <= 1e-5:
                std = 0.1

            if current_row.value * alpha > (avg + std) or current_row.value * alpha < (avg - std):
                # flag it as anomaly
                anomaly_list = anomaly_list.append(current_row)
            else:
                # remove the old data and append the new data
                window = window[1:]
                window = np.append(window, current_row.value)

        else:
            # append the data
            window = np.append(window, current_row.value)

    return anomaly_list


anomaly_list = HLPassFilter(6, CO2_pd, 1)


def composite_filter(half_window_size, df, lower_threshold, upper_threshold, n, lamda):
    # initialize the window
    window = np.array([])  # it only contains the value
    window_list = []  # it contain the whole information
    anomaly_list = pd.DataFrame()

    for row_index in range(len(df)):
        current_row = df.iloc[row_index]

        # the basic threshold filter is applied first
        if lower_threshold <= current_row.value <= upper_threshold:

            # then the second filter is applied
            if len(window) == half_window_size * 2 + 1:
                # current element
                current_element = window[half_window_size]
                # exclude the current element
                temp_array = np.delete(window, half_window_size)

                median = np.median(temp_array)
                std = temp_array.std()

                distance = np.linalg.norm(current_element - median)

                if distance >= n * std + lamda:
                    # flag it as anomaly
                    anomaly_list = anomaly_list.append(window_list[half_window_size])
                    # remove it from the list
                    window_list.pop(half_window_size)
                    window = np.delete(window, half_window_size)

                else:
                    # remove the old data and append the new data
                    window_list.pop(0)
                    window_list.append(current_row)

                    window = window[1:]
                    window = np.append(window, current_row.value)

            else:

                window_list.append(current_row)
                # append the data
                window = np.append(window, current_row.value)


        else:
            # flag it as anomaly
            anomaly_list = anomaly_list.append(current_row)

    return anomaly_list


anomaly_list = composite_filter(3, CO2_pd,
                                -40, 100,
                                0.5, 0.1)

# draw the picture
CDS_anaomaly_results = ColumnDataSource(data=anomaly_list)
CDS_CO2_results = ColumnDataSource(data=CO2_pd)

# visualization
unit_name = '6300c8'

time_interval = '5s-basis'
output_file("line.html")
# p = figure(title=unit_name+" "+time_interval,plot_width=800, plot_height=400,y_range=(0,1000))
# p.yaxis.ticker = [0, 200,400,600,800,1000]

p = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-20, 50))
p.yaxis.ticker = [-20, -10, 0, 10, 20, 30, 40, 50]

CO2_line = p.line(x='x', y='value',
                  source=CDS_CO2_results, line_width=2, color="#E63721")

# add legend
legend = Legend(items=[
    ("temperature", [CO2_line])
], location="center")
# add hover tips
CO2_hover = HoverTool(tooltips=[('x', '@x{%Y-%m-%d %H:%M:%S}'),
                                ('y', '@value')],
                      formatters={'@x': 'datetime'})

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

# add a circle renderer with a size, color, and alpha
CO2_point = p.circle(x='x', y='value',
                     color="navy", alpha=0.5,
                     source=CDS_anaomaly_results)

show(p)
