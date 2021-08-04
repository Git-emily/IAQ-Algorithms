# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:49:50 2021

@author: chens6
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def Mark_Data(data, file = 'marked_data.csv', save = False):
    
    mark_config = cfg()
    for new_col in mark_config['new_col'].unique():
        mark_infos = mark_config[mark_config['new_col'] == new_col]
        data[new_col] = 1
        for i in range(len(mark_infos)):
            mark_info = mark_infos.iloc[i]
            if mark_info['2nd_col'] in data.columns:
                bm_para = data[mark_info['2nd_col']].shift(int(mark_info['2nd_shift']))
            else:
                bm_para = mark_info['2nd_col']
                try:
                    bm_para = float(bm_para)
                except:
                    pass
            if mark_info['boolean'] == 'include':
                data[new_col] = data[new_col]&(data[mark_info['1st_col']].astype('str').str.contains(bm_para))
            elif mark_info['boolean'] in ['>','>=','==','<','<=']:
                exec('data[new_col] = data[new_col]&(data[mark_info["1st_col"]].shift(int(mark_info["1st_shift"]))%s(bm_para))'%mark_info['boolean'])
            elif (mark_info['boolean'] != mark_info['boolean']) or (mark_info['boolean']=='='):
                data[new_col] = data[mark_info["1st_col"]].shift(int(mark_info["1st_shift"]))
            else:
                exec('data[new_col] = (data[mark_info["1st_col"]].shift(mark_info["1st_shift"])%s(bm_para))'%mark_info['boolean'])
                if data[new_col].dtype in [np.dtype('timedelta64[ns]'),np.dtype('<m8[ns]')]:
                    data[new_col] = pd.to_timedelta(data[new_col]).dt.seconds
            if mark_info['filter_by_marker'] == mark_info['filter_by_marker']:
                data['filter'] = False 
                for para in mark_info['filter_by_marker'].split(','):
                    data['filter'] = data['filter']|data[para]
                data = data[data['filter']]
                data = data.drop('filter',1)
    if save:
        data.to_csv(file)
        
    return data

def Rename_Data(data, file = 'rename_data.csv', save = False):
    
    rename_config = cfg(filename = 'rename_config', clms = ('standard_name', 'column_name'))
    rename = {}
    for i in range(len(rename_config)):
        rename_info = rename_config.iloc[i]
        rename[rename_info['column_name']] = rename_info['standard_name']
    data = data[rename.keys()]
    
    data.replace(' - ',np.nan,inplace=True)
    for col in rename.keys():
        if col not in ['Serialnumber','DateTime','Cooling/Heating Select']:
            data[col] = data[col].astype('float')
    
    return data.rename(columns = rename)

def Bin_Data(data, file = 'bin_data.csv', save = False):
    
    bin_config = cfg(filename = 'bin_config',clms = ('parameter','start','end','interval'))
    for i in range(len(bin_config)):
        bin_info = bin_config.iloc[i]
        para = bin_info['parameter']
        if para in data.columns:
            start = bin_info['start']
            end = bin_info['end']
            interval = bin_info['interval']
            if start != start:
                #start = int(data[para].min()//interval*interval)
                start = round(float(data[para].min() // interval * interval),2)     # 5-20 change
            data[para + '_range'] = '%d~%d'%(data[para].min(),start)
            if end != end:
                #end = int((data[para].max()//interval+1)*interval)
                end = round(float((data[para].max() // interval + 1) * interval),2)     # 5-20 change            else:
            else:
                #end = int(((end - start) // interval) * interval + start)
                end = round(float(((end-start)//interval)*interval+start),2)        # 5-20 change
            data[para + '_range'][data[para] >= end] = '%d~%d'%(end,data[para].max())   # 5-20 change: out of if condition
            list_no = int((end - start) // interval + 1)    # 5-20 add
            if(str(interval).split('.')[1]=='0'):   # 5-20 add
                dec_no=0
            else:
                dec_no=len(str(interval)[str(interval).find('.') + 1:])
            if(dec_no==0):  # 5-20 add
                for i in range(int(start),int(end)+1,int(interval))[:-1]: #add int
                    data[para+'_range'][(data[para]>=i)&(data[para]<i+interval)] = '%d~%d'%(i,i+interval)
            else:
                for i in range(0,list_no,1):
                    data[para+'_range'][(data[para]>=round(start+interval*i,dec_no))&(data[para]<round(start+interval*i+interval,dec_no))] = '%s~%s' %(round(start+interval*i,dec_no),round(start+interval*i+interval,dec_no))
        else:
            print('bin_data KeyError: %s'%para)
    if save:
        data.to_csv(file)

    return data

def Miss_Data(data, THR = 60, time_col='datetime'):
    
    data[time_col] = pd.to_datetime(data[time_col])
    data['data_miss'] = pd.to_timedelta(data[time_col]-data[time_col].shift(1)).dt.seconds>60

    return data

def Get_Data(folder = 'data', file = 'data.csv', save = False):
    
    try:
        data = pd.read_csv(file,index_col=0)
        # data['datetime'] = pd.to_datetime(data.index)
    except:
        data = []
        for root,dirs,files in os.walk(folder):
            for fn in files:
                # data.append(pd.read_csv(os.path.join(root,fn),index_col=0))
                temp = pd.read_csv(os.path.join(root, fn), index_col=0)     #??file name
                temp['fn']=fn
                data.append(temp)
                i+=1
                print(i)
        data = pd.concat(data)
        if 'Date' in data.columns:
            if 'Time' in data.columns:
                data['datetime'] = pd.to_datetime(data['Date']+' '+data['Time'],format='%m/%d/%Y %I:%M:%S %p')
        if 'resolved_input_rvs_sense' in data.columns:
            if 'operating_mode' in data.columns:
                data['mode'] = ''
                data['mode'][(data.resolved_input_rvs_sense == 1) & (data.operating_mode == 1)] = 'cooling'
                data['mode'][(data.resolved_input_rvs_sense == 0) & (data.operating_mode == 0)] = 'heating'
                data['mode'][(data.resolved_input_rvs_sense) & (data.operating_mode == 0)] = 'defrost'

        # capdf = pd.read_csv('PEVunits.csv')
        # data['capacity'] = ''
        # data['sn'] = data['fn'].apply(lambda x: x.split('_')[1])
        # for sn in capdf.unit:
        #     data['capacity'][data.sn == sn] = capdf[capdf.unit == sn]['Capacity'].iloc[0]   #add capacity

        # 5-21 add
        NORCAP = {9: '24AC',
                  3: '24HP',
                  21: '36AC',
                  15: '36HP',
                  33: '48AC',
                  27: '48HP',
                  45: '60AC',
                  39: '60HP'}
        EC = {'CCN_COMFORT_AIRFLOW': 'Comfort',
              'CCN_EFFICIENCY_AIRFLOW': 'Efficiency'}
        data['model'] = ''
        data['operation'] = ''
        for j in NORCAP.keys():
            data['model'][data['Equipment Model'] == j] = NORCAP[j]
        for j in EC.keys():
            data['operation'][data['performance_mode_ccn'] == j] = EC[j]

        data['type'] = ''
        data['type'][data.model.str.contains('AC')] = 'AC'
        data['type'][data.model.str.contains('HP')] = 'HP'
        data['cap'] = ''
        data['cap'][data.model.str.contains('24')] = '2T'
        data['cap'][data.model.str.contains('36')] = '3T'
        data['cap'][data.model.str.contains('48')] = '4T'
        data['cap'][data.model.str.contains('60')] = '5T'

        if save:
            data.to_csv(file)
    
    return data

def Plot_Data(data,unit = '',result_folder = 'result'):
    
    try:
        os.mkdir(result_folder)
    except:
        pass
    sns.set_style('darkgrid')
    plt.rcParams['font.sans-serif'] = ['simhei','Arial']
    plt.rcParams['axes.unicode_minus'] = False
    # sns.set(context='talk',font='simhei')

    plot_config = cfg('plot_config',('plot_type','figsize','title','x','y','hue','col','row','plots_by','axhline','axvline','xlim','ylim','xticks_rotation','kwarg','grid_kwarg'))
    for i in range(len(plot_config)):
        plot_info = plot_config.iloc[i]
        
        if  plot_info['plots_by'] in data.columns:
            plot_by_values = np.sort(data[plot_info['plots_by']].unique())
            for value in plot_by_values:
                temp = data[data[plot_info['plots_by']] == value]
                Plot_Exec(temp,plot_info,unit)
                plt.title(unit + '%s (%s: %s)'%(plot_info['title'],plot_info['plots_by'], value))
                plt.savefig(os.path.join(result_folder,'%s%s(%s).png'%(unit, plot_info['title'], value)))
                plt.close()
        else:
            Plot_Exec(data,plot_info,unit)
            plt.savefig(os.path.join(result_folder,'%s%s.png'%(unit,plot_info['title'])))
            plt.close()

def Sort_Order(data,col):
    
    order_list = data[col].unique().tolist()
    order_list = [i for i in order_list if i==i]
    if '_range' in col:
        order_list.sort(key = lambda x: float(x.split('~')[0]))
        return str(order_list)
    else:
        return str(list(np.sort(order_list)))
        
def Plot_Exec(data, plot_info, unit):
    
    row = plot_info['row']
    col = plot_info['col']
    if plot_info['figsize'] == plot_info['figsize']:
        exec('plt.figure(figsize=(%s))'%plot_info['figsize'])
    if (row!=row)&(col!=col):
        plot_exec = 'sns.%s(data["%s"]'%(plot_info['plot_type'],plot_info['x'])
        for para in plot_info.index:
            if plot_info[para] == plot_info[para]:
                if para == 'kwarg':
                    plot_exec += ', %s'%plot_info[para]
                elif para in ['plot_type','figsize','title','col','row','axhline','axvline','xlim','ylim','xticks_rotation','plots_by','grid_kwarg']:
                    continue
                elif para == 'x':
                    if '_range' in plot_info[para]:
                        plot_exec += ' , order = '+Sort_Order(data,plot_info[para])
                elif para in ['y','hue']:
                    plot_exec += ', %s=data["%s"]'%(para,plot_info[para])
                    if para == 'hue':
                        plot_exec += ', hue_order = '+Sort_Order(data,plot_info[para])
                else:
                    plot_exec += ', %s="%s"'%(para,plot_info[para])
        exec(plot_exec+')')
        for para in ['title','axhline','axvline','xlim','ylim','xticks_rotation']:
            if plot_info[para] == plot_info[para]:
                if len(para.split('_'))==2:
                    exec('plt.%s(%s=%s)'%((para).split('_')[0],(para).split('_')[1],plot_info[para]))
                    if para == 'xticks_rotation':
                        if plot_info[para]%180 != 0:
                            plt.subplots_adjust(bottom=0.26)
                elif para == 'title':
                    exec('plt.%s("%s %s")'%(para,unit,plot_info[para]))
                elif 'line' in para:
                    exec("plt.%s(%s,ls='--',c='r')"%(para,plot_info[para]))
                else:
                    exec('plt.%s(%s)'%(para,plot_info[para]))
    else:
        grid_exec = 'g = sns.FacetGrid(data'
        for para in ['col','row','hue','grid_kwarg']:
            if plot_info[para] == plot_info[para]:
                if para == 'grid_kwarg':
                    plot_exec += ', %s'%plot_info[para]
                else:
                    grid_exec += ', %s = "%s"'%(para,plot_info[para])
                    grid_exec += ', %s_order = '%para + Sort_Order(data,plot_info[para])
        exec(grid_exec+',margin_titles=True,sharex=True,sharey=True)')
        plot_exec = 'g.map(sns.%s'%(plot_info['plot_type'])
        for para in plot_info.index:
            if plot_info[para] == plot_info[para]:
                if para == 'kwarg':
                    plot_exec += ', %s'%plot_info[para]
                elif para in ['plot_type','title','col','row','xlim','ylim','xticks_rotation','plots_by','grid_kwarg']:
                    continue
                elif para in ['x','y']:
                    plot_exec += ', "%s"'%(plot_info[para])
        if '_range' in plot_info['x']:
            plot_exec += ' , order = ' + Sort_Order(data,plot_info['x'])
        for para in ['xlim','ylim','xticks_rotation']:
            if plot_info[para] == plot_info[para]:
                if len(para.split('_'))==2:
                    exec('g.set_xticklabels(rotation=%d)'%(plot_info[para]))
                else:
                    exec('g.set(%s=(%s))'%(para,plot_info[para]))
        exec(plot_exec+')')
        
def cfg(filename = 'mark_config',clms = ('new_col','1st_col','1st_shift','boolean','2nd_col','2nd_shift','filter_by_marker'),eg = 'python'):
    
    try:
        config = pd.read_csv(filename+'.csv',engine= eg,encoding='utf_8_sig')
        try:
            del config['Unnamed: 0']
        except: pass
    except:
        print('No '+filename+' file available')
        config = pd.DataFrame(columns=clms)
        config.to_csv(filename+'.csv',index=False)
        
    return config

if __name__ == "__main__":

    data = Get_Data()
    data = Miss_Data(data)
    data = Mark_Data(data)
    data = Bin_Data(data)
    Plot_Data(data)        