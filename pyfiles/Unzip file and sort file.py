# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:19:18 2021

@author: wuyi1234
"""

import os
# import zipfile
import shutil


def make_zip(target_folder, folder_to_be_archived):
    shutil.make_archive(base_name=folder_to_be_archived,
                        format='zip',
                        root_dir=target_folder,
                        base_dir=folder_to_be_archived)
    # since the archived file would be saved under current working dir so move to the target folder
    shutil.move('{filename}.{filetype}'.format(filename=folder_to_be_archived, filetype='zip'), target_folder)


# make_zip(r'C:\Users\wuyi1234\Desktop\New\0101_0102','datalogger61a3fa(2021.01.01-2021.01.02)')


# dir_path=r'C:\Users\wuyi1234\Desktop\New\0109_0110'
dir_path = r'E:\0206_0301'
date = dir_path.split('\\')[-1]
start_date = '2021.' + date.split('_')[0][0:2] + '.' + date.split('_')[0][2:]
end_date = '2021.' + date.split('_')[1][0:2] + '.' + date.split('_')[1][2:]
full_date = start_date + "-" + end_date

destination = r'C:\Users\wuyi1234\Desktop\DataLogger\target unit'

for folder_name in os.listdir(dir_path):
    folder_path = os.path.join(dir_path, folder_name)
    # in case of unit name longer (e.g.datalogger63009b(2021.01.02-2021.01.03))
    new_folder_name = folder_name[0:16] + "(" + full_date + ")"
    # print(folder_path)
    # print(os.path.join(dir_path,new_folder_name))
    # rename
    os.rename(folder_path, os.path.join(dir_path, new_folder_name))
    # zip the folder
    make_zip(dir_path, new_folder_name)
    # distributed the file(zip and folder) to the corresponding folder
    if not os.path.exists(os.path.join(destination, folder_name[10:16], new_folder_name)):
        shutil.move(os.path.join(dir_path, new_folder_name),
                    os.path.join(destination, folder_name[10:16]))
    if not os.path.exists(os.path.join(destination, folder_name[10:16], '{0}.{1}'.format(new_folder_name, 'zip'))):
        shutil.move(os.path.join(dir_path, '{0}.{1}'.format(new_folder_name, 'zip')),
                    os.path.join(destination, folder_name[10:16]))

# shutil.move(r'C:\Users\wuyi1234\Desktop\New\0101_0102\datalogger61a52b(2021.01.01-2021.01.02)',
#              r'C:\Users\wuyi1234\Desktop')

# os.path.exists()
# os.path.join(r'C:\Users\wuyi1234\Desktop\New\0101_0102',
#              'datalogger61a52b(2021.01.01-2021.01.02)',
#              '')
