# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:46:24 2021

@author: WuYi
"""
import os
import shutil
import tempfile
import zipfile

path = "C:/Users/wuyi1234/Desktop/DataLogger/target unit/6300c5"
file_list = os.listdir(path)
# create a temporary dir for extracting and renaming all the zipped file
temp_dir = tempfile.TemporaryDirectory(dir=path)
temp_dir_name = temp_dir.name

for file in file_list:
    name = os.path.splitext(file)[0]
    suffix = os.path.splitext(file)[1]
    # then zip
    if suffix == ".zip":
        with zipfile.ZipFile(os.path.join(path, file), 'r') as zip_ref:
            zip_ref.extractall(temp_dir_name)
            # rename
            os.rename(os.path.join(temp_dir_name, os.listdir(temp_dir_name)[0]), os.path.join(temp_dir_name, name))
            shutil.move(os.path.join(temp_dir_name, name), path)
temp_dir.cleanup()
