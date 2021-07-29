# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:27:25 2021

@author: wuyi1234
"""

# dir_path should contain all the data needed for plotting
dir_path = r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa'

import datetime
import os

import pandas as pd

starting_hour = 16  # which is 24-hour format
execute_once = True

# first step is to find the time interval which files cover through
for file in os.listdir(dir_path):
    CurrFile = pd.read_csv(os.path.join(dir_path, file))
    # drop the last three digits in date (UTC)
    CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
    # convert your timestamps to datetime and then use matplotlib
    CurrFile["date-format"] = CurrFile["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
    # sort timestamp in ascending order
    CurrFile = CurrFile.sort_values(by='date-format', ascending=True)
    # delete the NA rows
    CurrFile = CurrFile.dropna()

    if execute_once:
        min_time = min(CurrFile['date-format'])
        max_time = max(CurrFile['date-format'])
        execute_once = False

    if min(CurrFile['date-format']) < min_time:
        min_time = min(CurrFile['date-format'])
    if max(CurrFile['date-format']) > max_time:
        max_time = max(CurrFile['date-format'])

# each element in file_list would be one day-data
file_list = []
# each element in date_list would be corresponding to weekday of the 24-hour
# len(date_list) should be equal to the file_list
date_list = []
time_list = []

starting_date = pd.Timestamp(min_time.year, min_time.month, min_time.day,
                             starting_hour, 0, 0)

ending_date = starting_date + pd.Timedelta(days=1)

while ending_date < max_time + pd.Timedelta(days=1):
    temp_df = pd.DataFrame()
    for file in os.listdir(dir_path):  # since 24-hour data may be contained in two files or more
        CurrFile = pd.read_csv(os.path.join(dir_path, file))
        # drop the last three digits in date (UTC)
        CurrFile["date (UTC)"] = CurrFile["date (UTC)"].map(lambda x: x[:-4])
        # convert your timestamps to datetime and then use matplotlib
        CurrFile["date-format"] = CurrFile["date (UTC)"].map(
            lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
        # sort timestamp in ascending order
        CurrFile = CurrFile.sort_values(by='date-format', ascending=True)
        # delete the NA rows
        CurrFile = CurrFile.dropna()

        filtered_file = CurrFile[(starting_date <= CurrFile["date-format"]) &
                                 (CurrFile["date-format"] < ending_date)]
        if len(filtered_file) != 0:
            temp_df = temp_df.append(filtered_file)

    if len(temp_df) != 0:  # there could be some missing day
        temp_df = temp_df.sort_values(by='date-format', ascending=True)
        # remove duplicate rows if there is
        temp_df = temp_df.drop_duplicates()
        # GMT to local time
        temp_df['date-format'] = temp_df['date-format'] - pd.Timedelta(hours=5)
        file_list.append(temp_df)

        if starting_date.weekday() in [0, 1, 2, 3]:
            date_list.append('weekday')
        else:
            date_list.append('weekend')

        time_list.append([starting_date, ending_date])

    starting_date += pd.Timedelta(days=1)
    ending_date += pd.Timedelta(days=1)

    print(starting_date)
    print(ending_date)

"""
SAX method
"""
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation
from itertools import compress
import numpy as np

weekday_lines = list(compress(file_list, np.array(date_list) == "weekday"))
weekend_lines = list(compress(file_list, np.array(date_list) == "weekend"))

# use raw data instead of moving average data
X_collection = []
# consider the first 20 lines
for i in range(len(weekday_lines) - 15):
    # step1 normalization
    X = weekday_lines[i]['value'].copy()
    X = np.array(X)
    X = X - np.mean(X)
    X = X / np.std(X)
    X = X.reshape((1, -1))
    # print(X.shape)
    paa = PiecewiseAggregateApproximation(window_size=12 * 60 * 1)
    X_paa = paa.transform(X)
    X_collection.append(X)

    print(i)
    if i == 0:
        X_paa_collection = X_paa
    else:
        X_paa_collection = np.vstack((X_paa_collection, X_paa))

# check normality
# import statsmodels.api as sm
# import pylab as py
# Z=weekday_lines[0]['value'].copy()
# Z=np.array(Z)
# Z=Z-np.mean(Z)
# Z=Z/np.std(Z) 

# sm.qqplot(Z, line ='45')
# py.show()


# then SAX transformation
sax = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
X_sax = sax.fit_transform(X_paa_collection)

# count
count_pd = pd.DataFrame(np.zeros((3, 24)),
                        columns=["BIN1", "BIN2", "BIN3", "BIN4", "BIN5", "BIN6",
                                 "BIN7", "BIN8", "BIN9", "BIN10", "BIN11", "BIN12",
                                 "BIN13", "BIN14", "BIN15", "BIN16", "BIN17", "BIN18",
                                 "BIN19", "BIN20", "BIN21", "BIN22", "BIN23", "BIN24"],
                        index=['a', 'b', 'c'])

for i in range(X_sax.shape[1]):
    column_unique = np.unique(X_sax[:, i], return_counts=True)
    print(column_unique[0])
    print(column_unique[1])
    print(count_pd.columns[i])
    count_pd.at[column_unique[0], count_pd.columns[i]] = column_unique[1]

# visualization
# plot the PPA results
# show the first line and its transform
import matplotlib.pyplot as plt

for j in range(len(X_collection)):
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(X_collection[j].flatten())),
             X_collection[j].flatten(), color='blue')
    for i, (y_value, y_sax) in enumerate(zip(X_paa_collection[j], X_sax[j])):
        plt.hlines(y=y_value, xmin=12 * 60 * 1 * i, xmax=12 * 60 * 1 * (i + 1), color='orange')
        plt.annotate(y_sax, xy=((12 * 60 * 1 * i + 12 * 60 * 2 * (i + 1)) / 2, y_value + 0.1), color='red', fontsize=20)
    plt.savefig(os.path.join(r'C:\Users\wuyi1234\Desktop\SAX', str(j + 1) + ".png"), format='png', dpi=600,
                bbox_inches='tight')
    plt.show()

# check the subsequence
# take the first 10 weekday data as the history data.
history = X_sax[0:10]
compare = X_sax[10:]
threshold = 3
window_size = 4


def all_subsequence(X, window_size):
    """
    X should be (#sample, sequence)
    it will return all the subsequnced defined by the window size
    """
    seq_list = []
    for i in range(X.shape[0]):
        curr_seq = X[i]
        for j in range(X.shape[1] - window_size + 1):
            seq_list.append(curr_seq[j:j + window_size])
    return seq_list


def hammingdistance(strA, strB):
    distance = 0
    if len(strA) != len(strB):
        print("two string must be equal length")
        return
    else:
        for i in range(len(strA)):
            if strA[i] != strB[i]:
                distance += 1
        return distance


history_seq = all_subsequence(history, window_size)

# no overlapping

anamoly_seq = {}
for j in range(compare.shape[0]):
    curr_seq = compare[j]
    for i in range(compare.shape[1] - window_size + 1):
        sub_seq = curr_seq[i:i + window_size]
        min_distance = 10000
        # compare sub_seq with all possible subsequence of history and find the nearest neighbor
        for k in history_seq:
            dist = hammingdistance(sub_seq, k)
            if dist < min_distance:
                min_distance = dist
                nearest_seq = k
                # print(min_distance,sub_seq,"(",j,i,")",k)
        if min_distance >= 3:
            # (j,i) is the location information
            anamoly_seq[(j, i, min_distance)] = (sub_seq, nearest_seq)
