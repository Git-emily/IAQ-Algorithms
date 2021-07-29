# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:20:43 2021

@author: WuYi
"""
from datetime import datetime

import numpy as np
import pandas as pd

# read file
CO2_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2/WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv"
CO2_fille_pd = pd.read_csv(CO2_file_path)
# drop the last three digits in date (UTC)
CO2_fille_pd["date (UTC)"] = CO2_fille_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
CO2_fille_pd["date-format"] = CO2_fille_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
CO2_fille_pd = CO2_fille_pd[["value", "date-format"]]
# rename the column
CO2_fille_pd.rename(columns={'value': "CO2"}, inplace=True)

PM1_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_PM1_PM1.0_CUBIC_IAQ_PM1/WuhanIAQ_PM1_PM1.0_CUBIC_IAQ_PM1_part2.csv"
PM1_fille_pd = pd.read_csv(PM1_file_path)
# drop the last three digits in date (UTC)
PM1_fille_pd["date (UTC)"] = PM1_fille_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
PM1_fille_pd["date-format"] = PM1_fille_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
PM1_fille_pd = PM1_fille_pd[["value", "date-format"]]
# rename the column
PM1_fille_pd.rename(columns={'value': "PM1"}, inplace=True)

PM10_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_PM10_PM10_CUBIC_IAQ_PM10/WuhanIAQ_PM10_PM10_CUBIC_IAQ_PM10_part2.csv"
PM10_fille_pd = pd.read_csv(PM10_file_path)
# drop the last three digits in date (UTC)
PM10_fille_pd["date (UTC)"] = PM10_fille_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
PM10_fille_pd["date-format"] = PM10_fille_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
PM10_fille_pd = PM10_fille_pd[["value", "date-format"]]
# rename the column
PM10_fille_pd.rename(columns={'value': "PM10"}, inplace=True)

PM2_5_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_PM25_PM2.5_CUBIC_IAQ_PM25/WuhanIAQ_PM25_PM2.5_CUBIC_IAQ_PM25_part2.csv"
PM2_5_fille_pd = pd.read_csv(PM2_5_file_path)
# drop the last three digits in date (UTC)
PM2_5_fille_pd["date (UTC)"] = PM2_5_fille_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
PM2_5_fille_pd["date-format"] = PM2_5_fille_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
PM2_5_fille_pd = PM2_5_fille_pd[["value", "date-format"]]
# rename the column
PM2_5_fille_pd.rename(columns={'value': "PM2.5"}, inplace=True)

humidity_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_RH_Humidity_CUBIC_IAQ_RH/WuhanIAQ_RH_Humidity_CUBIC_IAQ_RH_part2.csv"
humidity_file_pd = pd.read_csv(humidity_file_path)
# drop the last three digits in date (UTC)
humidity_file_pd["date (UTC)"] = humidity_file_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
humidity_file_pd["date-format"] = humidity_file_pd["date (UTC)"].map(
    lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
humidity_file_pd = humidity_file_pd[["value", "date-format"]]
# rename the column
humidity_file_pd.rename(columns={'value': "humidity"}, inplace=True)

temp_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_Temp_Temperature_CUBIC_IAQ_T/WuhanIAQ_Temp_Temperature_CUBIC_IAQ_T_part2.csv"
temp_file_pd = pd.read_csv(temp_file_path)
# drop the last three digits in date (UTC)
temp_file_pd["date (UTC)"] = temp_file_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
temp_file_pd["date-format"] = temp_file_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
temp_file_pd = temp_file_pd[["value", "date-format"]]
# rename the column
temp_file_pd.rename(columns={'value': "temperature"}, inplace=True)

VOC_file_path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02/datalogger62ff02(2020.12.31-2020.12.31)/WuhanIAQ_VOC_VOC_CUBIC_IAQ_VOC/WuhanIAQ_VOC_VOC_CUBIC_IAQ_VOC_part2.csv"
VOC_file_pd = pd.read_csv(VOC_file_path)
# drop the last three digits in date (UTC)
VOC_file_pd["date (UTC)"] = VOC_file_pd["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
VOC_file_pd["date-format"] = VOC_file_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
VOC_file_pd = VOC_file_pd[["value", "date-format"]]
# rename the column
VOC_file_pd.rename(columns={'value': "VOC"}, inplace=True)

aggregated_data = CO2_fille_pd.merge(PM1_fille_pd, how='inner', on='date-format')
aggregated_data = aggregated_data.merge(PM10_fille_pd, how='inner', on='date-format')
aggregated_data = aggregated_data.merge(PM2_5_fille_pd, how='inner', on='date-format')
aggregated_data = aggregated_data.merge(humidity_file_pd, how='inner', on='date-format')
aggregated_data = aggregated_data.merge(temp_file_pd, how='inner', on='date-format')
aggregated_data = aggregated_data.merge(VOC_file_pd, how='inner', on='date-format')
# drop date-format column
aggregated_data = aggregated_data.drop(columns=["date-format"])

# if sum is 0, then add one to all the column
aggregated_data["VOC"] = aggregated_data["VOC"] + 1

# divide data into batches, each batch has 60 examples
bath_size = 60


def split_data(df, batchsize):
    bath_list = []
    num_batches = len(df) // batchsize
    for i in range(num_batches):
        bath_list.append(df[batchsize * i:batchsize * (i + 1)])
    return bath_list


bath_list = split_data(aggregated_data, bath_size)


def EntropyWeight(df):
    columnSum = df.sum(axis=0)
    print(columnSum)
    df = df / columnSum
    factor = -1 / np.log(len(df))
    Entropy = factor * ((df * np.log(df)).sum(axis=0))
    DiversityDegree = 1 - Entropy
    NormalizedWeight = DiversityDegree / DiversityDegree.sum()
    return NormalizedWeight


aaa = EntropyWeight(bath_list[0])
bbb = EntropyWeight(bath_list[3])
type(aaa)
ccc = EntropyWeight(aaa)
ccc = pd.DataFrame(aaa).transpose()
# columnSum=aggregated_data.sum(axis=0)
# aggregated_data=aggregated_data/columnSum
# factor= -1/np.log(len(aggregated_data))
# Entropy= factor*(aggregated_data *np.log(aggregated_data)).sum(axis=0)
# DiversityCriterion=1-Entropy
# NormalizedWeight=DiversityCriterion/DiversityCriterion.sum()

result = pd.DataFrame()
for batch in bath_list:
    NormalizedWeight = EntropyWeight(batch)
    NormalizedWeight = pd.DataFrame(NormalizedWeight).transpose()
    result = result.append(NormalizedWeight)

# draw the plot according to the percentage
import matplotlib.pyplot as plt

result["x"] = range(len(result))

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(result["x"], result["CO2"], label="CO2")
ax.plot(result["x"], result["PM1"], label="PM1")
ax.plot(result["x"], result["PM10"], label="PM10")
ax.plot(result["x"], result["PM2.5"], label="PM2.5")
ax.plot(result["x"], result["humidity"], label="humidity")
ax.plot(result["x"], result["temperature"], label="temperature")
ax.plot(result["x"], result["VOC"], label="VOC")

ax.legend(loc='upper center', bbox_to_anchor=(1.20, 1.03))
plt.show()
