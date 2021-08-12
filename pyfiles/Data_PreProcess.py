import os
import shutil
import tempfile
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd

unit_dic = {'61a3fa': 'Vandiver2','61a52b':'Blessing2','61a579':'Frink','62fef0':'Christensen','62fef1':'Miller','62fef8':'Rembold',
'62fefd':'Liang2','62ff20':'Gettum','62ff25':'Liang1','63005c':'McGurer','630091':'Russel','630092':'Mensa',
'630093':'Roller','630094':'Sotiri','6300c7':'Wenzlick','630116':'Lewis','63011e':'King','6300c5': 'Hemphill','6300c4': 'Finney',
'63011d': 'Jones','630198': 'Burns','6309ec': 'Lee','6300c8': 'Malespin','63009b': 'Vandiver2','62ff02': 'Blessing1','6300cc': 'Vandiver1'}

def unzip_folder(path):
    print('Unzip the folder')
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

def select_files(path):
    print('Sort files')
    folders =[]
    file_list = os.listdir(path)
    for key in unit_dic.keys():
        new_folder = PATH + '\\unit\\' + key
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        for folder in file_list:
            folder_name = path+'\\'+folder
            if os.path.isdir(folder_name):
                for item in os.listdir(folder_name):
                    if key in item:
                        # if not os.path.exists(os.path.join(new_folder,item)):
                        shutil.move(os.path.join(folder_name, item), new_folder)
                        folders.append(new_folder)
    print(folders)
    return folders

def sensor_list(folder_path,files_list):
    sensor_list = []
    for item_path in folder_path:
        for file_name in files_list:
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file_name in file:
                        item_PATH = os.path.join(PATH, file_name,item_path.split('\\')[-1])
                        if not os.path.exists(item_PATH):
                            os.mkdir(item_PATH)
                        os.rename(os.path.join(root,file),os.path.join(root,root.split('\\')[-2]+'_'+file))
                        shutil.copy(os.path.join(root,root.split('\\')[-2]+file), item_PATH)

        sensor_list.append(os.path.join(PATH, file_name))
    return sensor_list

def data_preprocess(PATH,sensor_data,unit,variable,starting_date_time,ending_date_time):
    item_info = pd.DataFrame()
    All_files = os.listdir(sensor_data)
    for file_path in All_files:
        # item_pd = item+'_pd'
        item_pd = pd.read_csv(os.path.join(sensor_data,file_path))
        item_pd["date (UTC)"] = item_pd["date (UTC)"].map(lambda x: x[:-7])
        # convert your timestamps to datetime and then use matplotlib
        item_pd["date-format"] = item_pd["date (UTC)"].map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M"))
        item_pd = item_pd[["date-format", "value"]]
        item_pd = item_pd.groupby(by='date-format',as_index=False).max()
        item_pd["date-format"] = item_pd["date-format"] - pd.Timedelta(hours=5)  # CDT = GMT-5, EDT=GMT-4
        item_pd = item_pd.dropna()
        item_pd = item_pd[(item_pd['date-format'] >= starting_date_time) &(item_pd['date-format'] <= ending_date_time)]
        item_pd.rename(columns={'value': variable}, inplace=True)
        item_pd['date'] = item_pd['date-format'].map(lambda x: str(x).split(' ')[0])
        item_info = item_info.append(item_pd)
    item_info = item_info.sort_values('date-format')
    item_info = item_info.set_index('date-format').asfreq('1min')
    item_info = item_info.fillna(method='bfill')
    item_info["Time"] = (item_info.index).map(lambda x: str(x).split(' ')[1])
    two_level_index_series = item_info.set_index(['date','Time'])[variable]
    new_intem_info = two_level_index_series.unstack()
    print('Done_Part1')

    analysis_data = os.path.join(PATH, 'Analysis_' + variable)
    if not os.path.exists(analysis_data):
        os.mkdir(analysis_data)
    data_csv = str(starting_date_time).split(' ')[0] +'_'+ str(ending_date_time).split(' ')[0]
    new_intem_info.to_csv(os.path.join(analysis_data, data_csv+'.csv'))
    print('Done_Path',analysis_data)

if __name__ == '__main__':
    PATH = os.path.abspath(os.path.dirname(os.getcwd()))
    unit = '630094'
    starting_date_time = datetime.strptime('01.01.2021 00:00:00',"%d.%m.%Y %H:%M:%S")
    ending_date_time = datetime.strptime('14.01.2021 23:59:00',"%d.%m.%Y %H:%M:%S")
    files_list = ['WuhanIAQ_CO2', 'WuhanIAQ_PM25', 'WuhanIAQ_VOC']
    df_final = unzip_folder(PATH + '\\Raw_data')
    folder_path = select_files(PATH + '\\Raw_data')
    sensor_list = sensor_list(folder_path,files_list)
    data_preprocess(PATH,os.path.join(PATH,files_list[0],unit), unit,files_list[0].split('_')[1],starting_date_time,ending_date_time)