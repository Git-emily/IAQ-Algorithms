# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:17:27 2021

@author: WuYi1234
"""

import datetime
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation


# File1=pd.read_csv(r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa\datalogger630094(2021.01.02-2021.01.03)WuhanIAQ_CO2_CO2 concentration_CUBIC_IAQ_CO2_part2.csv')
# #drop the last three digits in date (UTC)
# File1["date (UTC)"] = File1["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# File1["date-format"] = File1["date (UTC)"].map(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# #sort timestamp in ascending order
# File1=File1.sort_values(by='date-format', ascending=True)
# #delete the NA rows
# File1=File1.dropna()


def fill_character(L, bitmap):
    """
        L is the length of side
        A | B 
        ------
        C | D
    
    """
    # top left
    # bitmap[0:L//2,0:L//2]+='a'
    bitmap[0:L // 2, 0:L // 2] = np.core.defchararray.add(bitmap[0:L // 2, 0:L // 2], 'a')
    if L // 2 != 1:
        fill_character(L // 2, bitmap[0:L // 2, 0:L // 2])

    # top right
    # bitmap[0:L//2,L//2:]+='b'
    bitmap[0:L // 2, L // 2:] = np.core.defchararray.add(bitmap[0:L // 2, L // 2:], 'b')
    if L // 2 != 1:
        fill_character(L // 2, bitmap[0:L // 2, L // 2:])

    # bottom left
    # bitmap[L//2:,0:L//2]+='c'
    bitmap[L // 2:, 0:L // 2] = np.core.defchararray.add(bitmap[L // 2:, 0:L // 2], 'c')
    if L // 2 != 1:
        fill_character(L // 2, bitmap[L // 2:, 0:L // 2])

    # bottom right
    # bitmap[L//2:,L//2:]+='d'
    bitmap[L // 2:, L // 2:] = np.core.defchararray.add(bitmap[L // 2:, L // 2:], 'd')
    if L // 2 != 1:
        fill_character(L // 2, bitmap[L // 2:, L // 2:])

    return bitmap


def CGR_representation(FileName, Level, T, W, bins_size, alphabet_size=4):
    """
    Level: the desired level of recursion of bitmap(usually 2 or 3)
    T:one-d time series data
    W:sliing window size
    alphabet_size: how many characters to be used in SAX, which is set for 4 and no need change
    bins_size: how many equally size bins 
    """
    # storing the sax sequence
    SAX_list = []
    for p in range(len(T) - W + 1):
        print(p)
        sliding_window = T[p:(p + W)]
        # then PPA
        # do the normalization
        # X should be (n_samples, n_timestamps) as input for PAA and SAX
        X = sliding_window.copy()
        X = np.array(X)
        X = X - np.mean(X)
        X = X / np.std(X)
        X = X.reshape((1, -1))
        # PAA transformation
        paa = PiecewiseAggregateApproximation(window_size=int(W / bins_size))
        X_paa = paa.transform(X)

        # then SAX transformation  
        sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal')
        X_sax = sax.fit_transform(X_paa)

        char = ''.join(X_sax[0][:])
        SAX_list.append(char)

    # get the possible combination of characters
    matching_str = {"".join(x): 0 for x in itertools.product("abcd", repeat=Level)}

    # do the frequence counting
    for SAX_seq in SAX_list:
        for sub_str in matching_str:
            matching_str[sub_str] += SAX_seq.count(sub_str)
    # do the normalization
    denominator = max(matching_str.values())
    for key in matching_str:
        matching_str[key] = matching_str[key] / denominator

    bitmap = np.full((2 ** Level, 2 ** Level), '', dtype='U{}'.format(Level))
    bitmap = fill_character(2 ** Level, bitmap)

    freqmap = np.zeros((2 ** Level, 2 ** Level))

    for key in matching_str:
        freqmap[np.where(bitmap == key)[0], np.where(bitmap == key)[1]] = matching_str[key]

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(freqmap, cmap='jet')
    fig.savefig(r"C:\Users\wuyi1234\Desktop\heatmaps\{}.png".format(FileName),
                bbox_inches='tight', pad_inches=0)
    return


# CGR_representation('sdfsdfsf',3,File1['value'],12*60*2,8)


dir = r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa'
for file in os.listdir(dir):
    curr_file = pd.read_csv(os.path.join(dir, file))
    # drop the last three digits in date (UTC)
    curr_file["date (UTC)"] = curr_file["date (UTC)"].map(lambda x: x[:-4])
    # convert your timestamps to datetime and then use matplotlib
    curr_file["date-format"] = curr_file["date (UTC)"].map(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
    # sort timestamp in ascending order
    curr_file = curr_file.sort_values(by='date-format', ascending=True)
    # delete the NA rows
    curr_file = curr_file.dropna()

    file_name = file[0:39]

    CGR_representation(file_name, 3, curr_file['value'], 12 * 60 * 2, 8)

# bitmap=np.full((8, 8), '', dtype='U10')
# freqmap=np.zeros((8,8))


# fill_character(8,bitmap)

# for key in matching_str:
#     freqmap[np.where(bitmap==key)[0],np.where(bitmap==key)[1]]=matching_str[key]


# fig, ax = plt.subplots()
# im = ax.imshow(freqmap,cmap='jet')   
# fig.savefig(r"C:\Users\wuyi1234\Desktop\heatmaps\test.png")

# plt.imshow(freqmap, cmap='jet')

# a = np.random.random((16, 16))
# plt.imshow(a, cmap='hot', interpolation='nearest')
# plt.show()


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

"""
