# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:31:35 2021

@author: wuyi1234
"""

from bokeh.io import  output_file,show
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter,Legend,HoverTool,ColumnDataSource
from bokeh.layouts import column
#from datetime import datetime
import pandas as pd 
import os
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pythermalcomfort.models import pmv

PATH = os.path.abspath(os.path.dirname(os.getcwd()))
warnings.filterwarnings("ignore", category=DeprecationWarning)


def Fahrenheit2Celsius(Fahrenheit):
    return (Fahrenheit-32)*(5/9)
    
    
def ThermalComfort(temperature,humidity):
    """
    temperature value in Fahrenheit
    relative humidity value in % 
    
    Returns
    -------
    thermal comfort
    """
    celsius=Fahrenheit2Celsius(temperature)
    pmv_value=pmv(tdb=celsius,tr=celsius,vr=0.2,rh=humidity,met=1.1,clo=0.6,
               standard='ISO', units='SI')
    if abs(pmv_value)<=1:
        pmv_score= abs(pmv_value)*(-30)+105
        pmv_score=np.min([pmv_score,100])
    if abs(pmv_value)>1:
        pmv_score=abs(pmv_value)*(-15)+90
        pmv_score=np.max([pmv_score,0])
    return pmv_score

def Ventilation(CO2):
    """
    CO2 value in ppm
    
    Returns
    -------
    ventilation score
    """
    if CO2<=900:
        ventilation_score=CO2*(-0.1)+165
        ventilation_score=np.min([ventilation_score,100])
    if CO2>900:
        ventilation_score=CO2*(-0.15)+210
        ventilation_score=np.max([ventilation_score,0])
    return ventilation_score
    
    
    
def AirQuality(PM25,VOC):
    """
    PM2.5 in ug/m3
    TVOC/VOC  in ug/m3
    """
    #calculate PM2.5 first 
    if PM25<=15:
        PM25_score=PM25*(-5)+150
        PM25_score=np.min([PM25_score,100])
    if PM25>15:
        PM25_score=PM25*(-0.75)+86.25
        PM25_score=np.max([PM25_score,0])
    
    #then calculate VOC
    VOC=round(VOC)
    if VOC==0:
    #it corresponds to 150 ug/m3
        VOC_score=150*(-0.05)+100  
        VOC_score=np.min([VOC_score,100])
    if VOC==1:
        VOC_score=650*(-0.01)+80
        VOC_score=np.max([VOC_score,0])
    if VOC==2:
        VOC_score=2000*(-0.01)+80
        VOC_score=np.max([VOC_score,0])
    if VOC==3:
        VOC_score=6500*(-0.01)+80
        VOC_score=np.max([VOC_score,0])
        
    return np.min([PM25_score,VOC_score])
        
    
    
    
def file_searching(folder_path,file_name):
    """
    Parameters
    ----------
    folder_path : the folder path which contains the file you want to search
    file_name : the file you want to search 

    Returns
    -------
    files_list : the list contains the absolute path pointing to the file you want to search

    """
    files_list=[]
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_name in file:
                files_list.append((os.path.join(root,file)))
    return files_list
    
    
import datetime
def date_time_generator(row):
    if  isinstance(row['Date'], str):
        
        year=int( row['Date'].split('/')[2])
        month=int(row['Date'].split('/')[0])
        day=int(row['Date'].split('/')[1])
    
        hour=row['Time'].hour
        minute=row['Time'].minute
        second=row['Time'].second
    
    else:
        year=row['Date'].year
        month=row['Date'].month
        day=row['Date'].day
        
        hour=row['Time'].hour
        minute=row['Time'].minute
        second=row['Time'].second
    
    return datetime.datetime(year,month,day,hour,minute,second)


def IAQIndexCalculation(IndoorTempHumidity_file_path,
                        folder_path, #which contains the CO2,PM2.5 and VOC files
                        CO2_file_name,
                        PM25_file_name,
                        VOC_file_name):
    
    #use indoor temperature and humidity as the main entity
    IndoorTempHumidity=pd.read_excel(IndoorTempHumidity_file_path,
                                     skiprows=5)
    
    #extract the key columns
    IndoorTempHumidity=IndoorTempHumidity[['Date','Time','Thermostat Temperature (F)','Thermostat Humidity (%RH)']]
    
    
    #clean the data 
    IndoorTempHumidity.dropna(axis=0,inplace=True)
    
    #create datetime variable
    IndoorTempHumidity['date_time']=IndoorTempHumidity.apply(lambda row:date_time_generator(row),axis=1)
    
    
    #find the relevent file and combine
    starting_date_time=min(IndoorTempHumidity['date_time'])
    ending_date_time=max(IndoorTempHumidity['date_time'])

    
    #adding prior and after "padding" (extend the original range)
    starting_date_time=starting_date_time-datetime.timedelta(minutes = 1)
    ending_date_time=ending_date_time+datetime.timedelta(minutes = 1)

    #find the other sensor data
    CO2_files_list=file_searching(folder_path,CO2_file_name)
    PM25_files_list=file_searching(folder_path,PM25_file_name)
    VOC_files_list=file_searching(folder_path,VOC_file_name)
    

    CO2_info =pd.DataFrame()
    for CO2_file_path in CO2_files_list:
        CO2_pd=pd.read_csv(CO2_file_path)
        #drop the last three digits in date (UTC)
        CO2_pd["date (UTC)"] = CO2_pd["date (UTC)"].map(lambda x: x[:-4])
        #convert your timestamps to datetime and then use matplotlib
        CO2_pd["date-format"] = CO2_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
        
        #drop the empty value
        CO2_pd=CO2_pd[["date-format","value"]]
        CO2_pd.rename(columns={'value': "CO2"},inplace=True)
        CO2_pd["date-format"]=CO2_pd["date-format"]-pd.Timedelta(hours=5)
        CO2_pd=CO2_pd.dropna()
        
        
        CO2_pd = CO2_pd[(CO2_pd['date-format'] >= starting_date_time) & 
                        (CO2_pd['date-format'] <= ending_date_time)]
       
        CO2_info=CO2_info.append(CO2_pd)

    
    PM25_info=pd.DataFrame()
    for PM25_file_path in PM25_files_list:
        PM25_pd=pd.read_csv(PM25_file_path)
        #drop the last three digits in date (UTC)
        PM25_pd["date (UTC)"] = PM25_pd["date (UTC)"].map(lambda x: x[:-4])
        #convert your timestamps to datetime and then use matplotlib
        PM25_pd["date-format"] = PM25_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
        
        #drop the empty value
        PM25_pd=PM25_pd[["date-format","value"]]
        PM25_pd.rename(columns={'value': "PM25"},inplace=True)
        PM25_pd["date-format"]=PM25_pd["date-format"]-pd.Timedelta(hours=5)
        PM25_pd=PM25_pd.dropna()
        
        PM25_pd = PM25_pd[(PM25_pd['date-format'] >= starting_date_time) & 
                        (PM25_pd['date-format'] <= ending_date_time)]
        
        PM25_info=PM25_info.append(PM25_pd)
        
        
    
    VOC_info=pd.DataFrame()         
    for VOC_file_path in VOC_files_list:
        VOC_pd=pd.read_csv(VOC_file_path)
        #drop the last three digits in date (UTC)
        VOC_pd["date (UTC)"] = VOC_pd["date (UTC)"].map(lambda x: x[:-4])
        #convert your timestamps to datetime and then use matplotlib
        VOC_pd["date-format"] = VOC_pd["date (UTC)"].map(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
        #drop the empty value
        VOC_pd=VOC_pd[["date-format","value"]]
        VOC_pd.rename(columns={'value': "VOC"},inplace=True)
        VOC_pd["date-format"]=VOC_pd["date-format"]-pd.Timedelta(hours=5)
        VOC_pd=VOC_pd.dropna()
        
        
        VOC_pd = VOC_pd[(VOC_pd['date-format'] >= starting_date_time) & 
                        (VOC_pd['date-format'] <= ending_date_time)]

        VOC_info=VOC_info.append(VOC_pd)

    ###############################################################################################
    #add CO2 column to the data
    #need to be sorted first by date-format
    CO2_info = CO2_info.sort_values('date-format')
    
    df_CO2=pd.merge_asof(IndoorTempHumidity, CO2_info['date-format'], 
                     left_on="date_time",right_on='date-format',
                     allow_exact_matches=True,direction='backward')
    
    df_CO2.rename(columns={"date-format": "CO2-date-format-backward"},inplace=True)
    
    
    df_CO2=pd.merge_asof(df_CO2, CO2_info['date-format'], 
                     left_on="date_time",right_on='date-format',
                     allow_exact_matches=True,direction='forward')
    
    df_CO2.rename(columns={"date-format": "CO2-date-format-forward"},inplace=True)
    #drop the NA rows 
    df_CO2=df_CO2.dropna()
    #shift one unit down
    df_CO2['CO2-date-format-forward'] = df_CO2['CO2-date-format-forward'].shift(1)
    #drop the NA rows 
    df_CO2=df_CO2.dropna()
    
    
    #calculate the average value over the past 5 minutes 
    CO2_avg=[]
    for i in range(len(df_CO2)):
        cur_row=df_CO2.iloc[i]
        forward_time=cur_row['CO2-date-format-forward']
        backward_time=cur_row['CO2-date-format-backward']
        
        diff_back_forward=(backward_time-forward_time).total_seconds()
        diff_back_now=(cur_row['date_time']-backward_time).total_seconds()
        
        if  270<=diff_back_forward<=330 and 0<=diff_back_now<=10:
            CO2_subset=CO2_info[(forward_time<=CO2_info['date-format']) & (CO2_info['date-format']<=backward_time)]
            avg_CO2=CO2_subset['CO2'].mean()
            print(avg_CO2)
        else:
            avg_CO2=np.NAN
        CO2_avg.append(avg_CO2)        
    df_CO2['CO2_avg']=CO2_avg
    
    
    #add PM25 column
    #need to be sorted first
    PM25_info = PM25_info.sort_values('date-format')
    
    df_PM25=pd.merge_asof(IndoorTempHumidity, PM25_info['date-format'], 
                     left_on="date_time",right_on='date-format',
                     allow_exact_matches=True,direction='backward')
    df_PM25.rename(columns={"date-format": "PM25-date-format-backward"},inplace=True)
    
    df_PM25=pd.merge_asof(df_PM25, PM25_info['date-format'], 
                     left_on="date_time",right_on='date-format',
                     allow_exact_matches=True,direction='forward')
    df_PM25.rename(columns={"date-format": "PM25-date-format-forward"},inplace=True)
    
    #drop the NA rows 
    df_PM25=df_PM25.dropna()
    #shift one unit down
    df_PM25['PM25-date-format-forward'] = df_PM25['PM25-date-format-forward'].shift(1)
    #drop the NA rows 
    df_PM25=df_PM25.dropna()
    
    #calculate the average value over the past 5 minutes 
    PM25_avg=[]
    for i in range(len(df_PM25)):
        cur_row=df_PM25.iloc[i]
        forward_time=cur_row['PM25-date-format-forward']
        backward_time=cur_row['PM25-date-format-backward']
        
        diff_back_forward=(backward_time-forward_time).total_seconds()
        diff_back_now=(cur_row['date_time']-backward_time).total_seconds()
        if  270<=diff_back_forward<=330 and 0<=diff_back_now<=10:
           PM25_subset=PM25_info[(forward_time<=PM25_info['date-format']) & (PM25_info['date-format']<=backward_time)]
           avg_PM25=PM25_subset['PM25'].mean()
           print(avg_PM25)
        else:
            avg_PM25=np.NAN
        PM25_avg.append(avg_PM25)   
    df_PM25['PM25_avg']=PM25_avg
    
    #add VOC colunm
    #need to be sorted first
    VOC_info = VOC_info.sort_values('date-format')
    df_VOC=pd.merge_asof(IndoorTempHumidity, VOC_info['date-format'], 
                     left_on="date_time",right_on='date-format',
                     allow_exact_matches=True,direction='backward')
    df_VOC.rename(columns={"date-format": "VOC-date-format-backward"},inplace=True)
    
    df_VOC=pd.merge_asof(df_VOC, VOC_info['date-format'], 
                     left_on="date_time",right_on='date-format',
                     allow_exact_matches=True,direction='forward')
    
    df_VOC.rename(columns={"date-format": "VOC-date-format-forward"},inplace=True)
    #drop the NA rows 
    df_VOC=df_VOC.dropna()
    #shift one unit down
    df_VOC['VOC-date-format-forward'] = df_VOC['VOC-date-format-forward'].shift(1)
    #drop the NA rows 
    df_VOC=df_VOC.dropna()
    
    #calculate the average value over the past 5 minutes 
    VOC_avg=[]
    for i in range(len(df_VOC)):
        cur_row=df_VOC.iloc[i]
        forward_time=cur_row['VOC-date-format-forward']
        backward_time=cur_row['VOC-date-format-backward']
        diff_back_forward=(backward_time-forward_time).total_seconds()
        diff_back_now=(cur_row['date_time']-backward_time).total_seconds()
       
        if  270<=diff_back_forward<=330 and 0<=diff_back_now<=10:
            VOC_subset=VOC_info[(forward_time<=VOC_info['date-format']) & (VOC_info['date-format']<=backward_time)]
            avg_VOC=VOC_subset['VOC'].mean()
            print(avg_VOC)
        else:
            avg_VOC=np.NAN
        VOC_avg.append(avg_VOC)   
    df_VOC['VOC_avg']=VOC_avg
    
    
    
    #aggregate the result
    df_final=df_CO2.merge(df_PM25[["date_time","PM25-date-format-forward","PM25-date-format-backward","PM25_avg"]],
                          left_on="date_time", right_on="date_time")
    
    df_final=df_final.merge(df_VOC[["date_time","VOC-date-format-forward","VOC-date-format-backward","VOC_avg"]],
                            left_on="date_time", right_on="date_time")
    
    
    
    #then calculating the IAQ score
    #calculate the IAQ index
    df_final=df_final.dropna()
    df_final['thermal_comfort']=df_final.apply(lambda row: ThermalComfort(row['Thermostat Temperature (F)'],
                                                                          row['Thermostat Humidity (%RH)']),axis=1)
    df_final['ventilation']=df_final['CO2_avg'].map(lambda x:Ventilation(x))
    df_final['air_quality']=df_final.apply(lambda row: AirQuality(row['PM25_avg'],
                                                                  row['VOC_avg']),axis=1)
                                       
    df_final['overall_index']=df_final.apply(lambda row:row['thermal_comfort']*(1/3)
                                                        +row['ventilation']*(1/3)
                                                        +row['air_quality']*(1/3),axis=1)
    
    
    #divide the df_final into daily curves
    starting_hour=11 #it is 24-hour format
    min_time=min(df_final['date_time'])
    max_time=max(df_final['date_time'])
    #each element in file_list would be one day-data
    file_list=[]
    #each element in date_list would be corresponding to weekday of the 24-hour
    #len(date_list) should be equal to the file_list
    date_list=[]
    time_list=[]
    
    #the starter
    starting_date=pd.Timestamp(min_time.year, min_time.month, min_time.day, 
                           starting_hour,0,0)

    ending_date=starting_date+pd.Timedelta(days=1)
    
    
    while ending_date < max_time + pd.Timedelta(days=1):
        temp_df=pd.DataFrame()
        filtered_file=df_final[(starting_date <=df_final["date_time"]) & 
                                       (df_final["date_time"] < ending_date)]
        if len(filtered_file)!=0:
            filtered_file=filtered_file.sort_values(by='date_time', ascending=True)  
            #remove duplicate rows if there is 
            filtered_file = filtered_file.drop_duplicates()
            file_list.append(filtered_file)
        
            if starting_date.weekday() in [0,1,2,3]:
                date_list.append('weekday') 
            else:
                date_list.append('weekend')
                
            time_list.append([starting_date,ending_date])
            
        starting_date+=pd.Timedelta(days=1)
        ending_date+=pd.Timedelta(days=1)
        
        print(starting_date)
        print(ending_date)
    
    
    
    return df_final,file_list,date_list,time_list

EcobeeData_name = '\\indoor sensor\\EcobeeData\\Sotiri\\Hovardas Ecobee; 2021-01-09-to-2021-02-09.xlsx'
unit_name = '\\target unit(this is demo data)\\630094'

df_final,file_list,date_list,time_list=IAQIndexCalculation(PATH+EcobeeData_name,PATH+unit_name,
                                                             "WuhanIAQ_CO2",
                                                             "WuhanIAQ_PM25",
                                                             "WuhanIAQ_VOC")



# from sklearn.linear_model import LinearRegression 


# def average_line(multiple_line):
#     """
#     multiple_line should be a list of lines, in which each line should contain a series of point
#     """
#     reg=LinearRegression()
#     only_for_once=True
#     for line in multiple_line:
#         line['date_ordinal']=np.arange(1,len(line['redefined-timestamp'])+1)
#         if only_for_once:
#             min_time=min(line['redefined-timestamp'])
#             max_time=max(line['redefined-timestamp'])
#             only_for_once=False
        
#         if min(line['redefined-timestamp'])<min_time:
#             min_time=min(line['redefined-timestamp'])
#         if max(line['redefined-timestamp'])>max_time:
#             max_time=max(line['redefined-timestamp'])
#     #generate artificial x value(datetime) sequence
#     x_ts=np.arange(min_time,max_time,datetime.timedelta(minutes=5)).astype(datetime.datetime)
#     y_ts=[]
#     #then find the two nearst timestamp
#     for x in x_ts:
#         y_avg=[]
#         for line in multiple_line:
#             #find the nearest to the x
#             print(x)
#             #reset index in case of non-consecutive index when slicing row acorrding to index
#             line.reset_index(drop=True,inplace=True)
#             line['abs_diff']=(line['redefined-timestamp']-x).map(lambda diff:abs(diff.total_seconds()))
#             min_index=line['abs_diff'].idxmin()
#             if line.loc[min_index]['abs_diff']==0:
#                 y_avg.append(line.loc[min_index]['thermal_comfort'])
#             else:
#                 if min_index==min(line.index):   
#                     two_points=line.loc[[min_index,min_index+1]]
 
#                 elif min_index==max(line.index):
#                     two_points=line.loc[[min_index,min_index-1]]
                    
                    
#                 else:
#                     temp_df=line.loc[[min_index-1,min_index+1]]
#                     second_min_index=temp_df['abs_diff'].idxmin()
#                     two_points=line.loc[[min_index,second_min_index]]
                
#                 arr=np.array(two_points['date_ordinal'])
#                 arr=arr.reshape(-1,1)
#                 #fit linear regression
#                 reg.fit(arr,two_points['thermal_comfort'])
#                 #then predict
#                 #transform the x value into ordinal scale
#                 time_diff=abs((two_points.iloc[0]['redefined-timestamp']-two_points.iloc[1]['redefined-timestamp']).total_seconds())
                    
#                 if x>line.loc[min_index]['redefined-timestamp']:
#                     x_temp=line.loc[min_index]['date_ordinal']+(line.loc[min_index]['abs_diff']/time_diff)*abs(two_points.iloc[0]['date_ordinal']-two_points.iloc[1]['date_ordinal'])
                    
#                 elif x<line.loc[min_index]['redefined-timestamp']:
#                     x_temp=line.loc[min_index]['date_ordinal']-(line.loc[min_index]['abs_diff']/time_diff)*abs(two_points.iloc[0]['date_ordinal']-two_points.iloc[1]['date_ordinal'])
                
#                 x_temp=np.array(x_temp)
#                 x_temp=x_temp.reshape(-1,1)
                
#                 y_pred=reg.predict(x_temp)
#                 y_avg.append(y_pred[0])
#         #use the exponentially weighting
#         y_ts.append(np.mean(y_avg)) 
     
#     return x_ts,y_ts
    
    


# x_ts,y_ts=average_line(file_list)



# plt.plot(file_list[0]['date_ordinal'],file_list[0]['thermal_comfort'],'blue')
# plt.show()

# plt.plot(file_list[0]['date_time'],file_list[0]['thermal_comfort'],'r')
# plt.show()







"""
visualization
Overall
"""
CDS_results=ColumnDataSource(data=df_final) 
#visualization
unit_name='630091'

time_interval='5min-basis'
output_file("line.html")


p = figure(title=unit_name+" "+time_interval,plot_width=800, plot_height=400,y_range=(-5,100))
p.yaxis.ticker = [-5, 0, 25,50,75,100]    
    


thermal_comfort_line=p.line(x='date_time', y='thermal_comfort',
                            source=CDS_results, line_width=2,color="#E63721")

#add legend
legend = Legend(items=[
    ("thermal_comfort", [thermal_comfort_line])
    ], location="center")
#add hover tips
thermal_comfort_hover= HoverTool(tooltips=[("x","@date_time{%Y-%m-%d %H:%M:%S}"),
                                           ("y","@thermal_comfort")],
                         formatters={'@date_time': 'datetime'})

p.add_tools(thermal_comfort_hover)
p.add_layout(legend, 'right')

p.xaxis.formatter=DatetimeTickFormatter(
                    seconds=["%Y-%m-%d %H:%M:%S"],
                    hours=["%Y-%m-%d %H:%M:%S"],
                    days=["%Y-%m-%d %H:%M:%S"],
                    months=["%Y-%m-%d %H:%M:%S"],
                    years=["%Y-%m-%d %H:%M:%S"],
                    )
p.xaxis.major_label_orientation = pi/4
p.xgrid.grid_line_color = None


############################################################
p1 = figure(title=unit_name+" "+time_interval,plot_width=800, plot_height=400,y_range=(-5,100))
p1.yaxis.ticker = [-5, 0, 25,50,75,100]

ventilation_line=p1.line(x='date_time', y='ventilation',
                source=CDS_results, line_width=2,color="#21E6CF")

#add legend
legend = Legend(items=[
    ("ventilation_line" , [ventilation_line])
    ], location="center")
#add hover tips
ventilation_hover= HoverTool(tooltips=[("x","@date_time{%Y-%m-%d %H:%M:%S}"),
                           ("y","@ventilation")],
                         formatters={'@date_time': 'datetime'})


p1.add_tools(ventilation_hover)
p1.add_layout(legend, 'right')

p1.xaxis.formatter=DatetimeTickFormatter(
                    seconds=["%Y-%m-%d %H:%M:%S"],
                    hours=["%Y-%m-%d %H:%M:%S"],
                    days=["%Y-%m-%d %H:%M:%S"],
                    months=["%Y-%m-%d %H:%M:%S"],
                    years=["%Y-%m-%d %H:%M:%S"],
                    )
p1.xaxis.major_label_orientation = pi/4
p1.xgrid.grid_line_color = None

#############################################################
p2 = figure(title=unit_name+" "+time_interval,plot_width=800, plot_height=400,y_range=(-5,100))
p2.yaxis.ticker = [-5,0, 25,50,75,100]


air_quality_line=p2.line(x='date_time', y='air_quality',
                source=CDS_results, line_width=2,color="#60EB10")

#add legend
legend = Legend(items=[
    ("air_quality_line" , [air_quality_line])
    ], location="center")

#add hover tips
air_quality_hover= HoverTool(tooltips=[("x","@date_time{%Y-%m-%d %H:%M:%S}"),
                           ("y","@air_quality")],
                         formatters={'@date_time': 'datetime'})


p2.add_tools(air_quality_hover)
p2.add_layout(legend, 'right')

p2.xaxis.formatter=DatetimeTickFormatter(
                    seconds=["%Y-%m-%d %H:%M:%S"],
                    hours=["%Y-%m-%d %H:%M:%S"],
                    days=["%Y-%m-%d %H:%M:%S"],
                    months=["%Y-%m-%d %H:%M:%S"],
                    years=["%Y-%m-%d %H:%M:%S"],
                    )
p2.xaxis.major_label_orientation = pi/4
p2.xgrid.grid_line_color = None



#################################################################
p3 = figure(title=unit_name+" "+time_interval,plot_width=800, plot_height=400,y_range=(-5,100))
p3.yaxis.ticker = [-5,0, 25,50,75,100]

overall_index_line=p3.line(x='date_time', y='overall_index',
                source=CDS_results, line_width=2,color="#EB10BC")

#add legend
legend = Legend(items=[
    ("overall_index_line" , [overall_index_line])
    ], location="center")


#add hover tips
overall_index_hover= HoverTool(tooltips=[("x","@date_time{%Y-%m-%d %H:%M:%S}"),
                           ("y","@overall_index")],
                         formatters={'@date_time': 'datetime'})


p3.add_tools(overall_index_hover)
p3.add_layout(legend, 'right')

p3.xaxis.formatter=DatetimeTickFormatter(
                    seconds=["%Y-%m-%d %H:%M:%S"],
                    hours=["%Y-%m-%d %H:%M:%S"],
                    days=["%Y-%m-%d %H:%M:%S"],
                    months=["%Y-%m-%d %H:%M:%S"],
                    years=["%Y-%m-%d %H:%M:%S"],
                    )
p3.xaxis.major_label_orientation = pi/4
p3.xgrid.grid_line_color = None


# put all the plots in a VBox
final_plot = column(p, p1,p2,p3)
show(final_plot)

"""
weekday and weekend pattern
"""
weekday_lines=[]
weekend_lines=[]
weekday_index=[]
weekend_index=[]
    
for i in range(len(date_list)):
    if date_list[i]=='weekday':
        weekday_index.append(i)
    if date_list[i]=='weekend':
        weekend_index.append(i)
            
for index in weekday_index:
    #change year,month,day to the same              
    mindate=time_list[index][0].date()
    file_list[index]['redefined-timestamp']=file_list[index]['date_time'].map(lambda x:x.replace(year=2000,month=1,day=1) 
                                            if x.date()==mindate else x.replace(year=2000,month=1,day=2))
    weekday_lines.append(file_list[index])
    
for index in weekend_index:
    mindate=time_list[index][0].date()    
    file_list[index]['redefined-timestamp']=file_list[index]['date_time'].map(lambda x:x.replace(year=2000,month=1,day=1) 
                                            if x.date()==mindate else x.replace(year=2000,month=1,day=2))
    weekend_lines.append(file_list[index])
    


plots_names=['thermal_comfort','ventilation','air_quality','overall_index']
color_palette=['r','c','g','m']
for col,color in zip(plots_names,color_palette):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(19,9))
    for line in weekday_lines:
        ax1.plot(line["redefined-timestamp"],line[col],color)
        #ax1.yaxis.set_ticks([0,10,20,30,40,50,60,70,80,90,100])
         
    for line in weekend_lines:
        ax2.plot(line["redefined-timestamp"],line[col],color)
        #ax2.yaxis.set_ticks([0,10,20,30,40,50,60,70,80,90,100])
        
    hrlocator=mdates.HourLocator()
    majorFmt = mdates.DateFormatter('%H:%M') 
        
    ax1.xaxis.set_major_locator(hrlocator)
    ax1.xaxis.set_major_formatter(majorFmt)
        
    ax2.xaxis.set_major_locator(hrlocator)
    ax2.xaxis.set_major_formatter(majorFmt)
    #rotate 90 degrees
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)   
    ax1.set_ylabel('weekday_{}'.format(col),fontsize=13)
    ax2.set_ylabel('weekend_{}'.format(col),fontsize=13)

    #save prior to show
    file_name=unit_name+col+".svg"
    if not os.path.exists(PATH + r'\export_IAQCalcu'):
        os.mkdir(PATH + r'\export_IAQCalcu')
    plt.savefig(os.path.join(PATH + r'\export_IAQCalcu', file_name), format='svg', dpi=1200, bbox_inches='tight')
        
    plt.show()


"""
histogram
"""
#overall
plots_names=['thermal_comfort','ventilation','air_quality','overall_index']
color_palette=['#E63721','#21E6CF','#60EB10','#EB10BC']
for col,color in zip(plots_names,color_palette):
    plt.hist(df_final[col], bins=[0,10,20,30,40,50,60,70,80,90,100],color=color)
    plt.xticks(range(0,110,10))
    plt.title(unit_name+'_'+col)
    file_name=unit_name+col+"_overall_hist.svg"
    if not os.path.exists(PATH + r'\export_IAQCalcu'):
        os.mkdir(PATH + r'\export_IAQCalcu')
    plt.savefig(os.path.join(PATH + r'\export_IAQCalcu', file_name), format='svg', dpi=1200, bbox_inches='tight')
    plt.show()
    


#weekday & weekend
weekday_accumulation=pd.DataFrame()
weekend_accumulation=pd.DataFrame()
for line in weekday_lines:
    weekday_accumulation=weekday_accumulation.append(line)
        
for line in weekend_lines:
    weekend_accumulation=weekend_accumulation.append(line)



for col,color in zip(plots_names,color_palette):
    fig,(ax1,ax2)=plt.subplots(2,1)

    ax1.hist(weekday_accumulation[col],bins=[0,10,20,30,40,50,60,70,80,90,100],color=color)    
    ax2.hist(weekend_accumulation[col],bins=[0,10,20,30,40,50,60,70,80,90,100],color=color)
    
    ax1.set_xticks(range(0,110,10))
    ax2.set_xticks(range(0,110,10))
    
    ax1.set_ylabel('weekday',fontsize=13)
    ax2.set_ylabel('weekend',fontsize=13)
    
    ax2.set_xlabel(col,fontsize=13)

    #save prior to show
    file_name=unit_name+col+"_hist.svg"
    if not os.path.exists(PATH + r'\export_IAQCalcu'):
        os.mkdir(PATH + r'\export_IAQCalcu')
    plt.savefig(os.path.join(PATH + r'\export_IAQCalcu', file_name), format='svg', dpi=1200, bbox_inches='tight')
        
    plt.show()

# if __name__ == '__main__':
#     EcobeeData_name = '\\indoor sensor\\EcobeeData\\Sotiri\\Hovardas Ecobee; 2021-01-09-to-2021-02-09.xlsx'
#     unit_name = '\\target unit(this is demo data)\\630094'
#     files_list = {'CO2_files_list': 'WuhanIAQ_CO2','PM25_file_list' : 'WuhanIAQ_PM25','VOC_file_list' : 'WuhanIAQ_VOC'}
#     time_interval = '5min-basis'
#
#     df_final, file_list, date_list, time_list = IAQIndexCalculation(PATH  + EcobeeData_name,PATH + unit_name, files_list)
#     Overall_visualization(df_final, file_list, date_list, time_list, unit_name, time_interval)
