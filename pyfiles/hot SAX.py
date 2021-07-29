# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:33:27 2021

@author: WuYi
"""
import datetime
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation


# File1=pd.read_csv(r'C:\Users\wuyi1234\Desktop\DataLogger\aaaa\datalogger630094(2021.01.01-2021.01.02)WuhanIAQ_PM25_PM2.5_CUBIC_IAQ_PM25_part2.csv')
# #drop the last three digits in date (UTC)
# File1["date (UTC)"] = File1["date (UTC)"].map(lambda x: x[:-4])
# #convert your timestamps to datetime and then use matplotlib
# File1["date-format"] = File1["date (UTC)"].map(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y %H:%M:%S"))
# #sort timestamp in ascending order
# File1=File1.sort_values(by='date-format', ascending=True)
# #delete the NA rows
# File1=File1.dropna()


def date_time_generator(row):
    if isinstance(row['Date'], str):

        year = int(row['Date'].split('/')[2])
        month = int(row['Date'].split('/')[0])
        day = int(row['Date'].split('/')[1])

        hour = row['Time'].hour
        minute = row['Time'].minute
        second = row['Time'].second

    else:
        year = row['Date'].year
        month = row['Date'].month
        day = row['Date'].day

        hour = row['Time'].hour
        minute = row['Time'].minute
        second = row['Time'].second

    return datetime.datetime(year, month, day, hour, minute, second)


File1 = pd.read_excel(
    r'C:\Users\wuyi1234\Desktop\DataLogger\indoor sensor\EcobeeData\DavidRussell\Russell Ecobee ereport-521750808280-2021-01-17-to-2021-02-17.xlsx',
    skiprows=5)

# extract the key columns
File1 = File1[['Date', 'Time', 'Thermostat Temperature (F)', 'Thermostat Humidity (%RH)']]
File1.dropna(axis=0, inplace=True)

File1['date-format'] = File1.apply(lambda row: date_time_generator(row), axis=1)

# visualize
plt.figure(figsize=(20, 7))
plt.plot(File1['date-format'], File1['Thermostat Temperature (F)'])
plt.show()


def BruteForceDiscordDiscory(T, W):
    """
    Arguments:
        T: one-d time series data
        W: sliding window size
    
    Return:
        best_so_far_dist: the largest distance of subsequence 
        best_so_far_loc: the start location of subsequence
    
    """
    best_so_far_dist = 0  # largest distance so far
    best_so_far_loc = -1

    for p in range(len(T) - W + 1):
        nearest_neighbor_dist = float('inf')
        for q in range(len(T) - W + 1):
            if abs(p - q) >= W:  # non-self match
                temp_dist = np.linalg.norm(T[p:(p + W)].array - T[q:(q + W)].array)
                print(q)
                if temp_dist < nearest_neighbor_dist:
                    nearest_neighbor_dist = temp_dist
        if nearest_neighbor_dist > best_so_far_dist:
            best_so_far_dist = nearest_neighbor_dist
            best_so_far_loc = p

    return best_so_far_dist, best_so_far_loc


# BruteForceDiscordDiscory(merged_File['value'],W=12*60*2)


def Outer_heuristic(T, W, alphabet_size=3, bins_size=12):
    # create a table like structure using pandas
    freq_table = pd.DataFrame(columns=['Characters', 'Frequency'])
    Char_column = []
    Freq_column = []
    for p in range(len(T) - W + 1):
        # do the normalization
        # X should be (n_samples, n_timestamps) as input for PAA and SAX
        X = T.copy().array[p:(p + W)]
        X = np.array(X)
        X = X - np.mean(X)
        if np.std(X) <= 0.25:
            char = 'b' * bins_size
        else:
            X = X / (np.std(X))
            X = X.reshape((1, -1))
            # PAA transformation
            paa = PiecewiseAggregateApproximation(window_size=int(W / bins_size))
            X_paa = paa.transform(X)

            # then SAX transformation  
            sax = SymbolicAggregateApproximation(n_bins=alphabet_size, strategy='normal')
            X_sax = sax.fit_transform(X_paa)

            char = ''.join(X_sax[0][:])
        Char_column.append(char)
    # then count the frequency
    unique_summary = Counter(Char_column)

    # print(unique_summary)
    for Char in Char_column:
        Freq_column.append(unique_summary[Char])

    freq_table['Characters'] = Char_column
    freq_table['Frequency'] = Freq_column

    return freq_table


# freq_table=Outer_heuristic(merged_File['value'][0:30],W=12)


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}

        # a list records the index of table
        # (if this node's is_end is True)
        self.index = []


class Trie(object):
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode("")

    def insert(self, word, index):
        """Insert a word into the trie"""
        # start from the root node
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node

        # Mark the end of a word
        node.is_end = True

        # add index location
        node.index.append(index)

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

    def dfs(self, node, prefix):
        """Depth-first traversal of the trie
        
        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.is_end:
            self.output.append((prefix + node.char, node.counter, node.index))

        for child in node.children.values():
            # recursively visit the rest nodes
            self.dfs(child, prefix + node.char)

    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of 
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list and end process
                return []

        # Traverse the trie to get all candidates
        # x[:-1] will leave out the last character
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)


# test
# t = Trie()
# t.insert("was")
# t.insert("word")
# t.insert("war")
# t.insert("what")
# t.insert("where")
# t.query("wh")
# t.query("why")
# [('what', 1), ('where', 1)]

# t = Trie()
# for i in range(len(freq_table[0:100])):
#     print(freq_table['Characters'][i])
#     t.insert(freq_table['Characters'][i],i)

# t.query('ca')

def Inner_heuristic(SearchStr, Trie, freq_table):
    first_to_vist = Trie.query(SearchStr)[0][2]
    # then the rest would random sequence
    index = np.arange(len(freq_table))
    index = np.delete(index, first_to_vist)
    index = list(index)
    random.shuffle(index)

    first_to_vist.extend(index)
    return first_to_vist


def non_overlapping(W, best_so_far_loc_ls, position):
    if len(best_so_far_loc_ls) == 0:
        return True
    else:
        return all(abs(np.array(best_so_far_loc_ls) - position) >= W)

    # aa=[15642]


# non_overlapping(W=12*60*2,best_so_far_loc_ls=aa,position=15642)

def HeuristicDiscordDiscrovery(T, W, Outer_heuristic, Inner_heuristic, Top_k=1):
    """
    Arguments:
        T: one-d time series data
        W: sliding window size
        Outer_heuristic: generate the order for outer loop
        Inner_heuristic: generate the order for inner loop
        Top_k: specify the top k most unusual subsequence to return
    Return:
        best_so_far_dist: the largest distance of subsequence for each top k most unusual subsequence 
        best_so_far_loc: the start location of top k most unusual subsequences
    """
    # best_so_far_dist=0
    # best_so_far_loc=-1

    best_so_far_dist_ls = []
    best_so_far_loc_ls = []

    # create two data structure(table, trie) for speeding up
    freq_table = Outer_heuristic(T, W, alphabet_size=3, bins_size=8)

    t = Trie()
    for i in range(len(freq_table)):
        print(freq_table['Characters'][i])
        # i is also the index as location
        t.insert(freq_table['Characters'][i], i)

    # then sort the Frequency column with increasing order
    freq_table = freq_table.sort_values("Frequency", ascending=True)

    for top_i in range(Top_k):
        j = 0
        best_so_far_dist = 0
        best_so_far_loc = -1

        for p in freq_table.index:
            print("top: " + str(top_i) + " position: " + str(p) + " Progress: " + str(j))
            nearest_neighbor_dist = float('inf')
            SearchStr = freq_table.loc[p]['Characters']

            inner_loop_seq = Inner_heuristic(SearchStr, t, freq_table)
            jump_inner_loop = False
            for q in inner_loop_seq:
                if abs(p - q) >= W and non_overlapping(W, best_so_far_loc_ls, p):  # non-self match

                    temp_dist = np.linalg.norm(T[p:(p + W)].array - T[q:(q + W)].array)

                    if temp_dist < best_so_far_dist:
                        print("-------------break out of inner loop at distance of {}------------------".format(
                            temp_dist))
                        jump_inner_loop = True
                        break
                    if temp_dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = temp_dist

            print('nearest_neighbor_dist: ' + str(nearest_neighbor_dist) +
                  ' best_so_far_dist: ' + str(best_so_far_dist) +
                  ' position: ' + str(p))

            if nearest_neighbor_dist > best_so_far_dist and jump_inner_loop == False and nearest_neighbor_dist != float(
                    'inf'):
                best_so_far_dist = nearest_neighbor_dist
                best_so_far_loc = p

                print('-----------------------------------------------------')
                print('updated best so far dist ' + str(best_so_far_dist))
                print('updated best so far location ' + str(best_so_far_loc))
                print('-----------------------------------------------------')

            j = j + 1
        # add result
        best_so_far_dist_ls.append(best_so_far_dist)
        best_so_far_loc_ls.append(best_so_far_loc)

    return best_so_far_dist_ls, best_so_far_loc_ls


best_so_far_dist_ls, best_so_far_loc_ls = HeuristicDiscordDiscrovery(File1['Thermostat Temperature (F)'], W=12 * 4,
                                                                     Outer_heuristic=Outer_heuristic,
                                                                     Inner_heuristic=Inner_heuristic, Top_k=1)

# plot to find the anomaly

# visualize
plt.figure(figsize=(20, 7))
plt.plot(File1['date-format'], File1['Thermostat Temperature (F)'], color='blue')
plt.plot(File1[best_so_far_loc_ls[0]:(best_so_far_loc_ls[0] + 12 * 4)]['date-format'],
         File1[best_so_far_loc_ls[0]:(best_so_far_loc_ls[0] + 12 * 4)]['Thermostat Temperature (F)'], color='red')
plt.plot(File1[best_so_far_loc_ls[1]:(best_so_far_loc_ls[1] + 12 * 4)]['date-format'],
         File1[best_so_far_loc_ls[1]:(best_so_far_loc_ls[1] + 12 * 4)]['Thermostat Temperature (F)'], color='green')
plt.show()

# best_loc = 3798 and best_dist=1799.1778678051817
