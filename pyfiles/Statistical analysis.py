# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:09:19 2021

@author: Wu Yi
"""

# #statiscticall test
# """
# merge the data into one
# """
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def FileMerger(FileName, FolderPath):
    df = pd.DataFrame()
    files_list = []
    for root, dirs, files in os.walk(FolderPath):
        for file in files:
            # print (os.path.join(root,file))
            # select only csv file
            _, file_extension = os.path.splitext(file)
            if file == FileName:
                files_list.append((file, os.path.join(root, file)))
                tempDf = pd.read_csv(os.path.join(root, file))
                df = df.append(tempDf)
    return files_list, df


# file name
BME280_RH = "BME280_i2c_humidity_Humidity_I2C_BME280_RH_part2.csv"
Wuhan_RH = "WuhanIAQ_RH_Humidity_CUBIC_IAQ_RH_part2.csv"
BME280_T = "BME280_i2c_temperature_Temperature_I2C_BME280_T_part2.csv"
Wuhan_T = "WuhanIAQ_Temp_Temperature_CUBIC_IAQ_T_part2.csv"

# """
# Unit:630918
# 1.	BME280_i2c_humidity_Humidity_I2C_BME280_RH VS WuhanIAQ_RH_Humidity_CUBIC_IAQ_RH
# 2.  BME280_i2c_temperature_Temperature_I2C_BME280_T vs. WuhanIAQ_Temp_Temperature_CUBIC_IAQ_T plot
# """
# """
# Relative humidity
# """
# #scatter plot
# #read data
# FolderPath="C:/Users/wuyi1234/Desktop/DataLogger/target unit/630198"
# _,U630918_BME280_RH=FileMerger(BME280_RH,FolderPath)
# _,U630918_Wuhan_RH=FileMerger(Wuhan_RH,FolderPath)

# U630918_BME280_RH=U630918_BME280_RH[["date (UTC)","value"]]
# U630918_Wuhan_RH=U630918_Wuhan_RH[["date (UTC)","value"]]

# #rename the column
# U630918_BME280_RH.rename(columns={'value': "BME280_RH"},inplace=True)
# U630918_Wuhan_RH.rename(columns={'value': "Wuhan_RH"},inplace=True)


# #drop the last three digits in date (UTC)
# U630918_BME280_RH["date (UTC)"] = U630918_BME280_RH["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# U630918_BME280_RH["date-format"] = U630918_BME280_RH["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U630918_BME280_RH=U630918_BME280_RH.drop(columns=["date (UTC)"])

# U630918_Wuhan_RH["date (UTC)"] = U630918_Wuhan_RH["date (UTC)"].map(lambda x: x[:-4])
# U630918_Wuhan_RH["date-format"] = U630918_Wuhan_RH["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U630918_Wuhan_RH=U630918_Wuhan_RH.drop(columns=["date (UTC)"])


# RH_Data=U630918_BME280_RH.merge(U630918_Wuhan_RH,how='inner', on='date-format')
# #same results
# #qqq=U630918_BME280_RH.merge(U630918_Wuhan_RH,how='inner', on='date (UTC)')

# #investigate the missing value after the merging
# # RH_Data_datatime=set(RH_Data["date-format"])
# # U630918_Wuhan_RH_datatime=set(U630918_Wuhan_RH["date-format"])

# # difference=U630918_Wuhan_RH_datatime-RH_Data_datatime
# # unmergedData1=U630918_Wuhan_RH.loc[ [x in difference for x in U630918_Wuhan_RH["date-format"]] ]

# #drop NA values
# RH_Data=RH_Data.dropna()

# plt.scatter(RH_Data.BME280_RH,RH_Data.Wuhan_RH,color="#95EB00",s=1)
# plt.xlabel("BME280_RH")
# plt.ylabel("Wuhan_RH")
# plt.savefig("C:/Users/wuyi1234/Desktop/DataLogger/U630918_RH.png", dpi=1200)
# #correlation
# corrM=RH_Data.corr()

# #do the statistical analysis
# #from scipy.stats import kstest
# #Kolmogorov-Smirnov test
# from scipy.stats import ks_2samp
# test_stat = ks_2samp(U630918_BME280_RH.dropna()["BME280_RH"], U630918_Wuhan_RH.dropna()["Wuhan_RH"])

# #two-sample
# import scipy.stats as stats
# stats.ttest_ind(RH_Data.BME280_RH,RH_Data.Wuhan_RH,equal_var=True)

# """
# Temperature
# """

# #scatter plot
# #read data
# FolderPath="C:/Users/wuyi1234/Desktop/DataLogger/target unit/630198"
# _,U630918_BME280_T=FileMerger(BME280_T,FolderPath)
# _,U630918_Wuhan_T=FileMerger(Wuhan_T,FolderPath)

# U630918_BME280_T=U630918_BME280_T[["date (UTC)","value"]]
# U630918_Wuhan_T=U630918_Wuhan_T[["date (UTC)","value"]]

# #rename the column
# U630918_BME280_T.rename(columns={'value': "BME280_T"},inplace=True)
# U630918_Wuhan_T.rename(columns={'value': "Wuhan_T"},inplace=True)


# #drop the last three digits in date (UTC)
# U630918_BME280_T["date (UTC)"] = U630918_BME280_T["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# U630918_BME280_T["date-format"] = U630918_BME280_T["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U630918_BME280_T=U630918_BME280_T.drop(columns=["date (UTC)"])

# U630918_Wuhan_T["date (UTC)"] = U630918_Wuhan_T["date (UTC)"].map(lambda x: x[:-4])
# U630918_Wuhan_T["date-format"] = U630918_Wuhan_T["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U630918_Wuhan_T=U630918_Wuhan_T.drop(columns=["date (UTC)"])


# RH_Data=U630918_BME280_T.merge(U630918_Wuhan_T,how='inner', on='date-format')

# #drop NA values
# RH_Data=RH_Data.dropna()

# plt.scatter(RH_Data.BME280_T,RH_Data.Wuhan_T,color="#95EB00",s=1)
# plt.xlabel("BME280_T")
# plt.ylabel("Wuhan_T")
# plt.savefig("C:/Users/wuyi1234/Desktop/DataLogger/U630918_Temperature.png", dpi=1200)
# #correlation
# corrM=RH_Data.corr()

# #do the statistical analysis
# #from scipy.stats import kstest
# #Kolmogorov-Smirnov test
# from scipy.stats import ks_2samp
# ks_2samp(U630918_BME280_T.dropna()["BME280_T"], U630918_Wuhan_T.dropna()["Wuhan_T"])

# #two-sample
# import scipy.stats as stats
# stats.ttest_ind(RH_Data.BME280_T,RH_Data.Wuhan_T,equal_var=True)


# """
# Unit:62ff02
# """
# """
# Relative humidity
# """
# #scatter plot
# #read data
# FolderPath="C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02"
# _,U62ff02_BME280_RH=FileMerger(BME280_RH,FolderPath)
# _,U62ff02_Wuhan_RH=FileMerger(Wuhan_RH,FolderPath)

# U62ff02_BME280_RH=U62ff02_BME280_RH[["date (UTC)","value"]]
# U62ff02_Wuhan_RH=U62ff02_Wuhan_RH[["date (UTC)","value"]]

# #rename the column
# U62ff02_BME280_RH.rename(columns={'value': "BME280_RH"},inplace=True)
# U62ff02_Wuhan_RH.rename(columns={'value': "Wuhan_RH"},inplace=True)


# #drop the last three digits in date (UTC)
# U62ff02_BME280_RH["date (UTC)"] = U62ff02_BME280_RH["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# U62ff02_BME280_RH["date-format"] = U62ff02_BME280_RH["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U62ff02_BME280_RH=U62ff02_BME280_RH.drop(columns=["date (UTC)"])

# U62ff02_Wuhan_RH["date (UTC)"] = U62ff02_Wuhan_RH["date (UTC)"].map(lambda x: x[:-4])
# U62ff02_Wuhan_RH["date-format"] = U62ff02_Wuhan_RH["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U62ff02_Wuhan_RH=U62ff02_Wuhan_RH.drop(columns=["date (UTC)"])


# RH_Data=U62ff02_BME280_RH.merge(U62ff02_Wuhan_RH,how='inner', on='date-format')


# #drop NA values
# RH_Data=RH_Data.dropna()

# plt.scatter(RH_Data.BME280_RH,RH_Data.Wuhan_RH,color="#95EB00",s=1)
# plt.xlabel("BME280_RH")
# plt.ylabel("Wuhan_RH")
# plt.savefig("C:/Users/wuyi1234/Desktop/DataLogger/U62ff02_RH.png", dpi=1200)
# #correlation
# corrM=RH_Data.corr()

# #do the statistical analysis
# #from scipy.stats import kstest
# #Kolmogorov-Smirnov test
# from scipy.stats import ks_2samp
# ks_2samp(U62ff02_BME280_RH.dropna()["BME280_RH"], U62ff02_Wuhan_RH.dropna()["Wuhan_RH"])

# #two-sample
# import scipy.stats as stats
# stats.ttest_ind(RH_Data.BME280_RH,RH_Data.Wuhan_RH,equal_var=True)

# """
# Temperature
# """
# #scatter plot
# #read data
# FolderPath="C:/Users/wuyi1234/Desktop/DataLogger/target unit/630198"
# _,U62ff02_BME280_T=FileMerger(BME280_T,FolderPath)
# _,U62ff02_Wuhan_T=FileMerger(Wuhan_T,FolderPath)

# U62ff02_BME280_T=U62ff02_BME280_T[["date (UTC)","value"]]
# U62ff02_Wuhan_T=U62ff02_Wuhan_T[["date (UTC)","value"]]

# #rename the column
# U62ff02_BME280_T.rename(columns={'value': "BME280_T"},inplace=True)
# U62ff02_Wuhan_T.rename(columns={'value': "Wuhan_T"},inplace=True)


# #drop the last three digits in date (UTC)
# U62ff02_BME280_T["date (UTC)"] = U62ff02_BME280_T["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# U62ff02_BME280_T["date-format"] = U62ff02_BME280_T["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U62ff02_BME280_T=U62ff02_BME280_T.drop(columns=["date (UTC)"])

# U62ff02_Wuhan_T["date (UTC)"] = U62ff02_Wuhan_T["date (UTC)"].map(lambda x: x[:-4])
# U62ff02_Wuhan_T["date-format"] = U62ff02_Wuhan_T["date (UTC)"].map(lambda x: datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# U62ff02_Wuhan_T=U62ff02_Wuhan_T.drop(columns=["date (UTC)"])


# RH_Data=U62ff02_BME280_T.merge(U62ff02_Wuhan_T,how='inner', on='date-format')

# #drop NA values
# RH_Data=RH_Data.dropna()

# plt.scatter(RH_Data.BME280_T,RH_Data.Wuhan_T,color="#95EB00",s=1)
# plt.xlabel("BME280_T")
# plt.ylabel("Wuhan_T")
# plt.savefig("C:/Users/wuyi1234/Desktop/DataLogger/U630918_Temperature.png", dpi=1200)
# #correlation
# corrM=RH_Data.corr()

# #do the statistical analysis
# #from scipy.stats import kstest
# #Kolmogorov-Smirnov test
# from scipy.stats import ks_2samp
# ks_2samp(U62ff02_BME280_T.dropna()["BME280_T"], U62ff02_Wuhan_T.dropna()["Wuhan_T"])

# #two-sample
# import scipy.stats as stats
# stats.ttest_ind(RH_Data.BME280_T,RH_Data.Wuhan_T,equal_var=True)


"""
use for loop for the rest calculation
"""
folderList = ["C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c4",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c5",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c8",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300cc",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/63009b",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/630198"]

unitList = ["62ff02",
            "6300c4",
            "6300c5",
            "6300c8",
            "6300cc",
            "63009b",
            "630198"]

# folderList=["C:/Users/wuyi1234/Desktop/DataLogger/target unit/6309ec"]
# unitList=["6309ec"]


# file name
BME280_RH_Name = "BME280_i2c_humidity_Humidity_I2C_BME280_RH_part2.csv"
Wuhan_RH_Name = "WuhanIAQ_RH_Humidity_CUBIC_IAQ_RH_part2.csv"
BME280_T_Name = "BME280_i2c_temperature_Temperature_I2C_BME280_T_part2.csv"
Wuhan_T_Name = "WuhanIAQ_Temp_Temperature_CUBIC_IAQ_T_part2.csv"

from collections import defaultdict

statResults = defaultdict(list)
import scipy.stats as stats
from scipy.stats import ks_2samp

for FolderPath, UnitName in zip(folderList, unitList):
    """
    Relative humidity
    """
    # scatter plot
    # read data
    print(BME280_RH_Name)
    print(UnitName)
    print(FolderPath)

    _, BME280_RH = FileMerger(BME280_RH_Name, FolderPath)
    _, Wuhan_RH = FileMerger(Wuhan_RH_Name, FolderPath)

    BME280_RH = BME280_RH[["date (UTC)", "value"]]
    Wuhan_RH = Wuhan_RH[["date (UTC)", "value"]]

    # rename the column
    BME280_RH.rename(columns={'value': "BME280_RH"}, inplace=True)
    Wuhan_RH.rename(columns={'value': "Wuhan_RH"}, inplace=True)

    # drop the last three digits in date (UTC)
    BME280_RH["date (UTC)"] = BME280_RH["date (UTC)"].map(lambda x: x[:-4])
    # convert your timestamps to datetime and then use matplotlib
    BME280_RH["date-format"] = BME280_RH["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
    BME280_RH = BME280_RH.drop(columns=["date (UTC)"])

    Wuhan_RH["date (UTC)"] = Wuhan_RH["date (UTC)"].map(lambda x: x[:-4])
    Wuhan_RH["date-format"] = Wuhan_RH["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
    Wuhan_RH = Wuhan_RH.drop(columns=["date (UTC)"])

    RH_Data = BME280_RH.merge(Wuhan_RH, how='inner', on='date-format')
    # drop NA values
    RH_Data = RH_Data.dropna()

    plt.scatter(RH_Data.BME280_RH, RH_Data.Wuhan_RH, color="#95EB00", s=1)
    plt.xlabel("BME280_RH")
    plt.ylabel("Wuhan_RH")

    fileName = UnitName + "_RH.png"
    plt.savefig("C:/Users/wuyi1234/Desktop/DataLogger/" + fileName, dpi=1200)
    # correlation
    corrM = RH_Data.corr()
    statResults[UnitName].append(corrM)

    # do the statistical analysis
    # from scipy.stats import kstest
    # Kolmogorov-Smirnov test

    # test_stat = ks_2samp(BME280_RH.dropna()["BME280_RH"], Wuhan_RH.dropna()["Wuhan_RH"])
    sample1 = BME280_RH.dropna()["BME280_RH"]
    sample1 = sample1 - sample1.mean()

    sample2 = Wuhan_RH.dropna()["Wuhan_RH"]
    sample2 = sample2 - sample2.mean()

    test_stat = ks_2samp(sample1, sample2)

    statResults[UnitName].append(test_stat)

    # two-sample

    test_stat = stats.ttest_ind(RH_Data.BME280_RH, RH_Data.Wuhan_RH, equal_var=True)
    statResults[UnitName].append(test_stat)

    """
    Temperature
    """
    # scatter plot
    # read data
    _, BME280_T = FileMerger(BME280_T_Name, FolderPath)
    _, Wuhan_T = FileMerger(Wuhan_T_Name, FolderPath)

    BME280_T = BME280_T[["date (UTC)", "value"]]
    Wuhan_T = Wuhan_T[["date (UTC)", "value"]]

    # rename the column
    BME280_T.rename(columns={'value': "BME280_T"}, inplace=True)
    Wuhan_T.rename(columns={'value': "Wuhan_T"}, inplace=True)

    # drop the last three digits in date (UTC)
    BME280_T["date (UTC)"] = BME280_T["date (UTC)"].map(lambda x: x[:-4])
    # convert your timestamps to datetime and then use matplotlib
    BME280_T["date-format"] = BME280_T["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
    BME280_T = BME280_T.drop(columns=["date (UTC)"])

    Wuhan_T["date (UTC)"] = Wuhan_T["date (UTC)"].map(lambda x: x[:-4])
    Wuhan_T["date-format"] = Wuhan_T["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
    Wuhan_T = Wuhan_T.drop(columns=["date (UTC)"])

    RH_Data = BME280_T.merge(Wuhan_T, how='inner', on='date-format')

    # drop NA values
    RH_Data = RH_Data.dropna()

    plt.scatter(RH_Data.BME280_T, RH_Data.Wuhan_T, color="#95EB00", s=1)
    plt.xlabel("BME280_T")
    plt.ylabel("Wuhan_T")

    fileName = UnitName + "_Temperature.png"

    plt.savefig("C:/Users/wuyi1234/Desktop/DataLogger/" + fileName, dpi=1200)
    # correlation
    corrM = RH_Data.corr()
    statResults[UnitName].append(corrM)

    # do the statistical analysis
    # from scipy.stats import kstest
    # Kolmogorov-Smirnov test

    test_stat = ks_2samp(BME280_T.dropna()["BME280_T"], Wuhan_T.dropna()["Wuhan_T"])
    statResults[UnitName].append(test_stat)

    # two-sample
    test_stat = stats.ttest_ind(RH_Data.BME280_T, RH_Data.Wuhan_T, equal_var=True)
    statResults[UnitName].append(test_stat)

#####################################################################################
folderList = ["C:/Users/wuyi1234/Desktop/DataLogger/target unit/62ff02",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c4",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c5",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c8",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300cc",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/63009b",
              "C:/Users/wuyi1234/Desktop/DataLogger/target unit/630198"]

unitList = ["62ff02",
            "6300c4",
            "6300c5",
            "6300c8",
            "6300cc",
            "63009b",
            "630198"]

for FolderPath, UnitName in zip(folderList, unitList):
    _, BME280_RH = FileMerger(BME280_RH_Name, FolderPath)
    _, Wuhan_RH = FileMerger(Wuhan_RH_Name, FolderPath)

    _, BME280_T = FileMerger(BME280_T_Name, FolderPath)
    _, Wuhan_T = FileMerger(Wuhan_T_Name, FolderPath)
    # output the merged csv

    BME280_RH.to_csv("C:/Users/wuyi1234/Desktop/DataLogger/merged target/" + UnitName + "BME280_RH.csv", index=False)
    Wuhan_RH.to_csv("C:/Users/wuyi1234/Desktop/DataLogger/merged target/" + UnitName + "Wuhan_RH.csv", index=False)

    BME280_T.to_csv("C:/Users/wuyi1234/Desktop/DataLogger/merged target/" + UnitName + "BME280_T.csv", index=False)
    Wuhan_T.to_csv("C:/Users/wuyi1234/Desktop/DataLogger/merged target/" + UnitName + "Wuhan_T.csv", index=False)
