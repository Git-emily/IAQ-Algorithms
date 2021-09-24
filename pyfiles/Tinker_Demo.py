# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:31:17 2021

@author: Wu Yi
"""
"""
GUI TKINTER
"""
import itertools
import os
from datetime import datetime
from math import pi
from tkinter import *
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column, gridplot
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper
from bokeh.models import DatetimeTickFormatter, Legend, Div, HoverTool, ColumnDataSource
from bokeh.plotting import figure


# create root widget
root = Tk()
root.title("Demo")
root.iconbitmap(os.getcwd() + "/carrier-logo.ico")
# root window size
root.geometry("800x400")
# window is not resizable in the X and Y direction
# root.resizable(False, False)


def LoadSelection():
    selections = AllFiles.curselection()
    # error indicator: 0- no error, 1-error
    err = -1
    if len(selections) == 0:
        # print ("no file selected")
        err = 1
        messagebox.showerror(title="Error", message="No file selected")
        return (err, None, None)
    else:
        err = 0
        filePath = []
        fileName = []
        for index in selections:
            filePath.append(AllFiles.get(index))

            # add date information to the name
            currFileName = os.path.split(AllFiles.get(index))[1]
            # get rid of the file suffix
            currFileName = currFileName.split(".")[0]

            CurrFile = pd.read_csv(AllFiles.get(index))

            # drop the last three digits in date (UTC)
            CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
            # convert your timestamps to datetime and then use matplotlib
            CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
            currFileName = currFileName + "(start:" + str(min(CurrFile["date-format"])) + " end:" + str(
                max(CurrFile["date-format"])) + ")"
            fileName.append(currFileName)

        return (err, fileName, filePath)


def FilePool():
    files = filedialog.askopenfilenames(title="add the file to file pool")
    # there could be multi-selections
    for file in files:
        AllFiles.insert(END, file)


# you can choose your favorite color
color = ("#E22E17", "#EEAD00", "#94C300", "#2A398D", "#26A7FF", "#7828FD", "#37DC94", "#F5B8D2")


def color_generator():
    yield from itertools.cycle(color)


# cyclic color palette
color_palette = color_generator()


def LinePlot():
    err, fileName, filePath = LoadSelection()
    if err == 0:
        # creat html file for data visualization
        output_file("TimeSeriesData.html")
        p = figure(plot_width=1600, plot_height=400,
                   toolbar_location="above",
                   tools='pan,xwheel_zoom,box_zoom,reset,save,zoom_in,zoom_out')
        p.add_layout(Legend(), "right")
        DivText = ""
        df_final = pd.DataFrame(columns=['xvalues', 'yvalues', 'names', 'colors'])

        for path, name, color in zip(filePath, fileName, color_palette):
            CurrFile = pd.read_csv(path)

            # count how many rows include the null values and delete the empty rows
            prev_len = len(CurrFile)
            CurrFile = CurrFile.dropna()
            cur_len = len(CurrFile)

            # print ("File {} has {} row(s) deleted".format(name,prev_len-cur_len))
            FileInfo = "File:{} has <b>{} row(s) deleted</b> </br>".format(name, prev_len - cur_len)
            DivText = DivText + FileInfo

            # drop the last three digits in date (UTC)
            CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
            # convert your timestamps to datetime and then use matplotlib
            CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

            # accumulating df and add up into a big df
            # p.line(CurrFile["date-format"],CurrFile["value"],legend_label=name,color=color)

            tempDf = pd.DataFrame(columns=['xvalues', 'yvalues', 'names', 'colors'])
            tempDf["xvalues"] = CurrFile["date-format"]
            tempDf["yvalues"] = CurrFile["value"]
            tempDf["names"] = name
            tempDf["colors"] = color
            df_final = df_final.append(tempDf, ignore_index=True)

        # plot the data
        group_list = df_final.names.unique()
        for i in range(len(group_list)):
            data = {'x': df_final.loc[df_final.names == group_list[i]].xvalues,
                    'y': df_final.loc[df_final.names == group_list[i]].yvalues,
                    'name': df_final.loc[df_final.names == group_list[i]].names,
                    'color': df_final.loc[df_final.names == group_list[i]].colors}
            source = ColumnDataSource(data=data)
            p.line(x='x', y='y',
                   source=source,
                   legend_label=group_list[i],
                   color=data['color'].unique()[0])

        # add hover tips
        hover = HoverTool(tooltips=[("group", "@name"),
                                    ("x", "@x{%Y-%m-%d %H:%M:%S}"),
                                    ("y", "@y")],
                          formatters={'@x': 'datetime'})

        p.add_tools(hover)

        FileInfoDiv = Div(text=DivText,
                          style={'font-size': '100%', 'color': 'black'})

        p.xaxis.formatter = DatetimeTickFormatter(
            seconds=["%Y-%m-%d %H:%M:%S"],
            hours=["%Y-%m-%d %H:%M:%S"],
            days=["%Y-%m-%d %H:%M:%S"],
            months=["%Y-%m-%d %H:%M:%S"],
            years=["%Y-%m-%d %H:%M:%S"],
        )
        p.xaxis.major_label_orientation = pi / 4
        p.legend.label_text_font_size = "8pt"
        # disable the logo
        p.toolbar.logo = None
        show(column(p, FileInfoDiv))
        # save(column(p,FileInfoDiv))
    return


# draw multiple line and tooltip
# from collections import defaultdict
# def LinePlot():
#     err,fileName,filePath=LoadSelection()
#     if err==0:
#         # creat html file for data visualization
#         output_file("TimeSeriesData.html")
#         p = figure(plot_width=1600, plot_height=400,
#                     toolbar_location="above",
#                     tools='pan,xwheel_zoom,box_zoom,reset,save,zoom_in,zoom_out')
#         p.add_layout(Legend(),"right")
#         DivText=""
#         DataCollection=defaultdict(list)
#         for path,name,color in zip(filePath,fileName,color_palette):
#             CurrFile=pd.read_csv(path)

#             #count how many rows include the null values and delete the empty rows
#             prev_len=len(CurrFile)
#             CurrFile=CurrFile.dropna()
#             cur_len=len(CurrFile)

#             #print ("File {} has {} row(s) deleted".format(name,prev_len-cur_len))
#             FileInfo="File:{} has <b>{} row(s) deleted</b> </br>".format(name,prev_len-cur_len)
#             DivText=DivText+FileInfo


#             #drop the last three digits in date (UTC)
#             CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
#             #convert your timestamps to datetime and then use matplotlib
#             CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))


#             DataCollection["xvalues"].append(CurrFile["date-format"])
#             DataCollection["yvalues"].append(CurrFile["value"])
#             DataCollection["lineColor"].append(color)
#             DataCollection["names"].append(name)


#         source = ColumnDataSource(DataCollection)
#         p.multi_line(xs='xvalues', ys='yvalues',
#                       legend="names",line_color='lineColor',
#                       source=source)
#         ##Add hover tools, basically an invisible line over the original line
#         #expand the DataCollection
#         df_final=pd.DataFrame(columns=['xvalues','yvalues','names'])
#         for xvalues,yvalues,names in zip(DataCollection["xvalues"],DataCollection["yvalues"],DataCollection["names"]):
#             tempDf = pd.DataFrame(columns=['xvalues','yvalues','names'])
#             tempDf["xvalues"]=xvalues
#             tempDf["yvalues"]=yvalues
#             tempDf["names"]=names
#             df_final=df_final.append(tempDf,ignore_index=True)


#         source2=ColumnDataSource(dict(
#                                 invisible_xs=df_final.xvalues,
#                                 invisible_ys=df_final.yvalues,
#                                 names = df_final.names))
#         line=p.line('invisible_xs',
#                     'invisible_ys',
#                     source=source2,
#                     alpha=0)
#         hover = HoverTool(tooltips =[
#                     ('name','@names'),
#                     ('x','@invisible_xs{%Y-%m-%d %H:%M:%S}'),
#                     ('y','@invisible_ys')],
#                     formatters={'@invisible_xs': 'datetime'})
#         hover.renderers = [line]
#         p.add_tools(hover)

#         FileInfoDiv = Div(text=DivText,
#                         style={'font-size': '100%', 'color': 'black'})

#         p.xaxis.formatter=DatetimeTickFormatter(
#             seconds=["%Y-%m-%d %H:%M:%S"],
#             hours=["%Y-%m-%d %H:%M:%S"],
#             days=["%Y-%m-%d %H:%M:%S"],
#             months=["%Y-%m-%d %H:%M:%S"],
#             years=["%Y-%m-%d %H:%M:%S"],
#             )
#         p.xaxis.major_label_orientation = pi/4
#         p.legend.label_text_font_size="8pt"
#         #disable the logo
#         p.toolbar.logo = None
#         show(column(p,FileInfoDiv))
#     return


def CorrelationPlot():
    err, fileName, filePath = LoadSelection()
    if err == 0:
        # define empty dataframe
        df = pd.DataFrame()

        DivText = ""
        # extract value column from different files
        for name, path in zip(fileName, filePath):

            CurrFile = pd.read_csv(path)

            # count how many rows include the null values and delete the empty rows
            prev_len = len(CurrFile)
            CurrFile = CurrFile.dropna()
            cur_len = len(CurrFile)

            # print ("File {} has {} row(s) deleted".format(name,prev_len-cur_len))
            FileInfo = "File:{} has <b>{} row(s) deleted</b> </br>".format(name, prev_len - cur_len)
            DivText = DivText + FileInfo

            # drop the last three digits in date (UTC)
            CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
            # convert your timestamps to datetime and then use matplotlib
            CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))

            CurrFile = CurrFile[["date-format", "value"]]

            if len(df) == 0:
                df[name] = CurrFile['value']
                df["date-format"] = CurrFile['date-format']
            # merge the data according to the date-format
            else:
                CurrFile.rename(columns={'value': name}, inplace=True)
                df = df.merge(CurrFile, how='inner', on='date-format')

        # drop the data-formate column
        df.drop(columns=['date-format'], inplace=True)
        if len(df) == 0:
            messagebox.showerror(title="Error", message="No data available for calculation")
            return

        else:
            # calculate the correlation matrix
            corrM = df.corr()

            # collapse the dataframe
            stackedCorrM = pd.DataFrame(corrM.stack(), columns=['CorreValue']).reset_index()
            # gradual color palette
            colors = ["#FF4D50", "#F15261", "#E35773", "#D45C84", "#C66196", "#B867A7", "#AA6CB9", "#9B71CA", "#8D76DC",
                      "#7F7BED"]
            mapper = LinearColorMapper(palette=colors, low=-1, high=1)

            variables1 = list(stackedCorrM['level_0'].unique())
            variables2 = list(stackedCorrM['level_1'].unique())

            output_file("Correlation Matrix.html")

            FileInfoDiv = Div(text=DivText,
                              style={'font-size': '100%', 'color': 'black'})

            p = figure(title="Correlation Matrix",
                       x_range=variables1, y_range=variables2,
                       x_axis_location="below",
                       plot_width=800, plot_height=800,
                       toolbar_location='above',
                       tools='pan,wheel_zoom,box_zoom,reset,save,zoom_in,zoom_out',
                       tooltips=[('x', '@level_0'),
                                 ('y', '@level_1'),
                                 ('value', '@CorreValue')])

            # erase the grid line
            p.grid.grid_line_color = None
            p.axis.axis_line_color = None
            p.axis.major_tick_line_color = None
            p.axis.major_label_text_font_size = "9px"
            p.axis.major_label_standoff = 0
            p.xaxis.major_label_orientation = pi / 3

            p.rect(x="level_0", y="level_1",  # location
                   width=1, height=1,  # rectangle shape
                   source=stackedCorrM,
                   fill_color={'field': 'CorreValue', 'transform': mapper},
                   line_color="black")

            color_bar = ColorBar(color_mapper=mapper,
                                 major_label_text_font_size="7px",
                                 ticker=BasicTicker(desired_num_ticks=len(colors)),
                                 label_standoff=6, border_line_color=None, location=(0, 0))
            p.add_layout(color_bar, 'right')
            # disable the logo
            p.toolbar.logo = None
            show(column(p, FileInfoDiv))
            return


def HistogramPlot():
    err, fileName, filePath = LoadSelection()
    if err == 0:
        output_file("histogram.html")
        # create an empty list
        plotList = [None] * len(filePath)
        i = 0
        DivText = ""
        for path, name in zip(filePath, fileName):
            CurrFile = pd.read_csv(path)

            # count how many rows include the null values and delete the empty rows
            prev_len = len(CurrFile)
            CurrFile = CurrFile.dropna()
            cur_len = len(CurrFile)
            print("File {} has {} row(s) are deleted".format(name, prev_len - cur_len))
            FileInfo = "File:{} has <b>{} row(s) deleted</b> </br>".format(name, prev_len - cur_len)
            DivText = DivText + FileInfo

            hist, edges = np.histogram(CurrFile['value'], bins=20)
            plotList[i] = figure(title=name,
                                 tools='pan,wheel_zoom,box_zoom,reset,save,zoom_in,zoom_out')
            plotList[i].quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
            i = i + 1
        FileInfoDiv = Div(text=DivText,
                          style={'font-size': '100%', 'color': 'black'})
        plotList.append(FileInfoDiv)
        show(gridplot(plotList, ncols=len(filePath), toolbar_options=dict(logo=None)))
    return


def ClearList():
    AllFiles.delete(0, END)
    return


filePoolFrame = LabelFrame(root, text="File Pool")
scrollbarY = Scrollbar(filePoolFrame, orient=VERTICAL)
scrollbarX = Scrollbar(filePoolFrame, orient=HORIZONTAL)

AllFiles = Listbox(filePoolFrame,
                   yscrollcommand=scrollbarY.set, xscrollcommand=scrollbarX.set,
                   width=70, height=10,
                   selectmode="multiple")
# config the scrollbar
scrollbarY.config(command=AllFiles.yview)
scrollbarX.config(command=AllFiles.xview)

dataAnalysisFrame = LabelFrame(root, text="Data Analysis")
lineButton = Button(dataAnalysisFrame, text="Line Plot", command=LinePlot, width=13)
correlationButton = Button(dataAnalysisFrame, text="Correlation Plot", command=CorrelationPlot, width=13)
histButton = Button(dataAnalysisFrame, text="Histogram Plot", command=HistogramPlot, width=13)

# img=ImageTk.PhotoImage(Image.open("C:/Users/wuyi1234/Desktop/DataLogger/carrier-logo.png"))
# myLabel=tk.Label(image=img)
selectionButton = Button(root, text="Select File(s)", command=FilePool, width=13)
quitButton = Button(root, text="Exit Program", command=root.destroy, width=13)
clearButton = Button(root, text="Clear All", command=ClearList, width=13)

# layout all the widgets(top-down sequence)
selectionButton.grid(row=0, column=0, sticky=W)

filePoolFrame.grid(row=1, column=0, sticky=N)
# sequence matters
scrollbarX.pack(side=BOTTOM, fill=X)
AllFiles.pack(side=LEFT, fill=Y)
scrollbarY.pack(side=RIGHT, fill=Y)

dataAnalysisFrame.grid(row=1, column=1, sticky=N)
lineButton.pack(fill=BOTH)
correlationButton.pack(fill=BOTH)
histButton.pack(fill=BOTH)

clearButton.grid(row=2, column=0, sticky=W)
quitButton.grid(row=2, column=1, sticky=W)

root.mainloop()

# debug the LinePlot


# output_file("1111.html")
# p = figure(plot_width=1600, plot_height=400,toolbar_location="above")
# p.add_layout(Legend(),"right")
# DivText=""

# CurrFile=pd.read_csv("C:/Users/wuyi1234/Desktop/DataLogger/new data/1110_1111/datalogger62ff02_1110_1111/IREF32A_Voltage_NET_CONC/IREF32A_Voltage_NET_CONC_part2.csv")

# #count how many rows include the null values and delete the empty rows
# prev_len=len(CurrFile)
# CurrFile.dropna()
# cur_len=len(CurrFile)


# FileInfo="File:{} has <b>{} row(s) deleted</b> </br>".format("name",prev_len-cur_len)
# DivText=DivText+FileInfo


# #drop the last three digits in date (UTC)
# CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# #CurrFile=CurrFile[0:8]

# p.line(CurrFile["date-format"],CurrFile["value"],legend_label="test",color="red")


# FileInfoDiv = Div(text=DivText,
#                       style={'font-size': '100%', 'color': 'black'})

# p.xaxis.formatter=DatetimeTickFormatter(
#             seconds=["%Y-%m-%d %H:%M:%S"],
#             hours=["%Y-%m-%d %H:%M:%S"],
#             days=["%Y-%m-%d %H:%M:%S"],
#             months=["%Y-%m-%d %H:%M:%S"],
#             years=["%Y-%m-%d %H:%M:%S"],
#             )
# p.xaxis.major_label_orientation = pi/4
# p.legend.label_text_font_size="8pt"

# show(column(p,FileInfoDiv))


# import os

# def ListAllFile(path):
#     """List all the csv files within a folder using absolute path

#     Args:
#          path: Path to folder containing csv files.

#     Returns:
#         file_list: all the files under that path
#     """
#     files_list=[]
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             #delete suffix "_part2.csv"
#             fileName=file[:-10]
#             #print (os.path.join(root,file))
#             files_list.append((fileName,os.path.join(root,file)))
#     return files_list
