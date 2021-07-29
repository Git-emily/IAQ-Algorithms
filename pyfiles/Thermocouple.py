# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:42:59 2021

@author: WuYi1234
"""
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
            if file == file_name:
                files_list.append((os.path.join(root, file)))
    return files_list


VOC_file_name = "Thermocouple_T_MCP9600_Temperature_TC_0X60_part2.csv"
folder_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c8"

VOC_list = file_searching(folder_path, VOC_file_name)

VOC = pd.DataFrame()
for VOC_file in VOC_list:
    temp_file = pd.read_csv(VOC_file)
    VOC = VOC.append(temp_file)

# VOC.to_csv (r'C:\Users\wuyi1234\Desktop\VOC.csv', index = False, header=True)


# preprocessing data
VOC["date (UTC)"] = VOC["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
VOC["x"] = VOC["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

VOC.columns

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter, Legend, HoverTool, ColumnDataSource
import pandas as pd
from math import pi

# visualize
CDS_results = ColumnDataSource(data=VOC)
# visualization
unit_name = '6300c8'

time_interval = '5s-basis'
output_file("line.html")
p = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(-20, 80))
p.yaxis.ticker = [-20, 0, 20, 40, 60, 80]

VOC_line = p.line(x='x', y='value',
                  source=CDS_results, line_width=2, color="blue")

# add legend
legend = Legend(items=[
    ("Temperature_TC_0X60", [VOC_line])
], location="center")
# add hover tips
VOC_hover = HoverTool(tooltips=[("x", "@x{%Y-%m-%d %H:%M:%S}"),
                                ("y", "@value")],
                      formatters={'@x': 'datetime'})

p.add_tools(VOC_hover)
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

show(p)

"""
"""
PM10_file_name = "WuhanIAQ_PM10_PM10_CUBIC_IAQ_PM10_part2.csv"
folder_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff20"

PM10_list = file_searching(folder_path, PM10_file_name)

PM10 = pd.DataFrame()
for PM10_file in PM10_list:
    temp_file = pd.read_csv(PM10_file)
    PM10 = PM10.append(temp_file)

# VOC.to_csv (r'C:\Users\wuyi1234\Desktop\VOC.csv', index = False, header=True)
PM10.to_csv(r'C:\Users\wuyi1234\Desktop\PM10.csv', index=False, header=True)

import datetime

# preprocessing data
PM10["date (UTC)"] = PM10["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
PM10["x"] = PM10["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

PM10.value

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter, Legend, HoverTool, ColumnDataSource
from datetime import datetime
import os
from math import pi

# visualize
CDS_results = ColumnDataSource(data=PM10)
# visualization
unit_name = '62ff20'

time_interval = '5s-basis'
output_file("line.html")
p = figure(title=unit_name + " " + time_interval, plot_width=800, plot_height=400, y_range=(0, 800))
p.yaxis.ticker = [0, 100, 200, 300, 400, 500, 600, 800]

PM10_line = p.line(x='x', y='value',
                   source=CDS_results, line_width=2, color="#E63721")

# add legend
legend = Legend(items=[
    ("PM10", [PM10_line])
], location="center")
# add hover tips
PM10_hover = HoverTool(tooltips=[("x", "@x{%Y-%m-%d %H:%M:%S}"),
                                 ("y", "@value")],
                       formatters={'@x': 'datetime'})

p.add_tools(PM10_hover)
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

show(p)
