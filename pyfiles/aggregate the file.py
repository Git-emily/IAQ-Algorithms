# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:09:06 2021

@author: wuyi1234
"""
import os
import shutil


# find and search the file
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
            if file_name in file:  # fuzzy match
                files_list.append((os.path.join(root, file)))
    return files_list


# create a new folder within the current directory
PATH = os.path.abspath(os.path.dirname(os.getcwd()))
folder_name = 'CO2_DATA'
folder_dir = os.path.join(PATH, folder_name)
if not os.path.exists(folder_dir):
    os.mkdir(folder_dir)  # create the folder

dir_path = PATH + r'\target unit(this is demo data)\630094'

files_list = file_searching(dir_path, 'WuhanIAQ_CO2')

for file_path in files_list:
    # copy and paste
    target_path = os.path.join(folder_dir, os.path.split(file_path)[1])
    shutil.copyfile(file_path, target_path)
    # rename
    prefix = file_path.split('\\')[-3]
    new_name = prefix + os.path.split(file_path)[1]
    os.rename(target_path, os.path.join(folder_dir, new_name))
