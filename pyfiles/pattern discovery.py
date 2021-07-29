# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:15:26 2021

@author: WuYi1234
"""

import datetime
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation

File1 = pd.read_csv(
    r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa\datalogger630094(2021.01.01-2021.01.02)WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv')
# drop the last three digits in date (UTC)
File1["date (UTC)"] = File1["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
File1["date-format"] = File1["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
# sort timestamp in ascending order
File1 = File1.sort_values(by='date-format', ascending=True)
# delete the NA rows
File1 = File1.dropna()

File2 = pd.read_csv(
    r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa\datalogger630094(2021.01.02-2021.01.03)WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv')
# drop the last three digits in date (UTC)
File2["date (UTC)"] = File2["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
File2["date-format"] = File2["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
# sort timestamp in ascending order
File2 = File2.sort_values(by='date-format', ascending=True)
# delete the NA rows
File2 = File2.dropna()

File3 = pd.read_csv(
    r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa\datalogger630094(2021.01.03-2021.01.04)WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv')
# drop the last three digits in date (UTC)
File3["date (UTC)"] = File3["date (UTC)"].map(lambda x: x[:-4])
# convert your timestamps to datetime and then use matplotlib
File3["date-format"] = File3["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
# sort timestamp in ascending order
File3 = File3.sort_values(by='date-format', ascending=True)
# delete the NA rows
File3 = File3.dropna()

plt.figure(figsize=(20, 7))
plt.plot(File1['date-format'], File1['value'], color='blue')
plt.show()

# first step is to divide the whole time series datai into 12 equal length bins(2-hour bin)
total_len = len(File1['value'])
num_bin = 12
indexs = np.array_split(np.arange(total_len), num_bin)

for index in indexs:
    plt.figure(figsize=(20, 7))
    plt.plot(File1['date-format'], File1['value'])
    plt.plot(File1['date-format'][index], File1['value'][index])
    plt.show()

# partition it into the several isolated part
bin_size_within_bin = 3
Char_column = []
for index in indexs:
    X = np.array(File2['value'][index])
    X_len = len(X)
    # do the sax transformation
    X = X - np.mean(X)
    X = X / (np.std(X))
    X = X.reshape((1, -1))
    # PAA transformation
    paa = PiecewiseAggregateApproximation(window_size=math.ceil(X_len / bin_size_within_bin))
    X_paa = paa.transform(X)

    # then SAX transformation  
    sax = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
    X_sax = sax.fit_transform(X_paa)
    char = ''.join(X_sax[0][:])
    Char_column.append(char)

freq_dict = Counter(Char_column)

# moving window
W = 12 * 60 * 2
T = File1['value']
SAX_list = []
for p in range((len(T) - W + 1)):
    sliding_window = T[p:(p + W)]
    X = sliding_window.copy()
    X = np.array(X)
    X = X - np.mean(X)
    X = X / np.std(X)
    X = X.reshape((1, -1))
    # PAA transformation
    paa = PiecewiseAggregateApproximation(window_size=math.ceil(W / 5))
    X_paa = paa.transform(X)

    # then SAX transformation  
    sax = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
    X_sax = sax.fit_transform(X_paa)

    char = ''.join(X_sax[0][:])
    SAX_list.append(char)

# count the frequence of each character
freq_dict = Counter(SAX_list)

# find the occurence of infrequent string
freq_index = [i for i, x in enumerate(SAX_list) if x == "bcbcb"]

for i in freq_index:
    plt.figure(figsize=(20, 7))
    plt.plot(File1['date-format'], File1['value'])
    plt.plot(File1['date-format'][i:i + W], File1['value'][i:i + W], color='red')
    plt.show()
    print(i)

# use the whole one-day time series
bin_size = 12
X = np.array(File1['value'][index])
X_len = len(X)
# do the sax transformation
X = X - np.mean(X)
X = X / (np.std(X))
X = X.reshape((1, -1))
# PAA transformation
paa = PiecewiseAggregateApproximation(window_size=math.ceil(X_len / bin_size))
X_paa = paa.transform(X)

# then SAX transformation  
sax = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
X_sax = sax.fit_transform(X_paa)

bin_size = 12
X = np.array(File3['value'][index])
X_len = len(X)
# do the sax transformation
X = X - np.mean(X)
X = X / (np.std(X))
X = X.reshape((1, -1))
# PAA transformation
paa = PiecewiseAggregateApproximation(window_size=math.ceil(X_len / bin_size))
X_paa = paa.transform(X)

# then SAX transformation  
sax = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
X_sax = sax.fit_transform(X_paa)
