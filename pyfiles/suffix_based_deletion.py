# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:14:04 2021

@author: wuyi1234
"""
import os


def suffix_based_deletion(folder_path, suffix_list):
    delerted_files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # print(file[-9:-4])  part12 would be extracted as
            # print(file)
            filename = os.path.splitext(file)[0]
            print(filename[filename.find('part'):])

            if filename[filename.find('part'):] in suffix_list:
                delerted_files_list.append((os.path.join(root, file)))
                os.remove(os.path.join(root, file))
    return delerted_files_list


aa = suffix_based_deletion("C:/Users/wuyi1234/Desktop/DataLogger/target unit/New folder/datalogger630198",
                           ["part1", "part2"])
