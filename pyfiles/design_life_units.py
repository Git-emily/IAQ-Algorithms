# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:52:17 2021

@author: chens6
"""

import pandas as pd
import datetime
from advanced_data_analytics_tool import Plot_Data
from uszipcode import SearchEngine

def Get_Data():
    try:
        alldata = pd.read_csv('all_units.csv')
    except:
        try:
            data = pd.read_csv('startdays.csv')
        except:
            NORCAP = {9:'24AC',
                      3:'24HP',
                      21:'36AC',
                      15:'36HP',
                      33:'48AC',
                      27:'48HP',
                      45:'60AC',
                      39:'60HP'}
            data = pd.read_parquet('all.gzip')
            data = data[data.NORCAP!=0]
            data = data[data.index>'2020-01-01 00:00:00']
            
            for unit in data.unit.unique():
                try:
                    start = min(data[(data.unit==unit)&(data.index>'2020-01-01 00:00:00')].index)
                    data = data[((data.unit==unit)&(data.index==start))|(data.unit!=unit)]
                    print(unit,len(data),start)
                except:
                    pass
            
            data['model'] = ''
            for model in NORCAP.keys():
                data['model'][data.NORCAP==model] = NORCAP[model]
            data.to_csv('startdays.csv')
        data.index = pd.to_datetime(data.index)
        data['days'] = data.index
        data['days'] = (datetime.datetime.now()-data['days']).dt.days
        data=data[data.days>365]
        
        # Plot_Data(data,'')
        
        ul = pd.read_csv('unitslocation.csv',header=None)
        unitlocation = {'unit':[],
                        'zipcode':[]}
        for i in range(len(ul)):
            try:
                if ('W' in ul.iloc[i,0]) & (len(ul.iloc[i+1,0])==5):
                    unitlocation['unit'].append(ul.iloc[i,0])
                    unitlocation['zipcode'].append(ul.iloc[i+1,0])
                    # unitlocation[ul.iloc[i,0]] = ul.iloc[i+1,0]
            except:
                pass
            
        unitlocation = pd.DataFrame.from_dict(unitlocation)
            
        search = SearchEngine(simple_zipcode=True)
        
        unitlocation['lat'] = unitlocation.zipcode.apply(lambda x: search.by_zipcode(str(x)).to_dict()['lat'])
        unitlocation['lng'] = unitlocation.zipcode.apply(lambda x: search.by_zipcode(str(x)).to_dict()['lng'])
        unitlocation.zipcode = unitlocation.zipcode.apply(lambda x: search.by_zipcode(str(x)).to_dict()['state'])
        
        
        data = pd.merge(data,unitlocation,on=['unit'],how='left')
        data['tonnage'] = data['model'].apply(lambda x: int(x[:2])/12)
        data['type'] = data['model'].apply(lambda x: x[2:])
        data['Sampling'] = 'All data'
        
        
        alldata = [data]
        for model in data.model.unique():
            temp = data[data.model==model]
            temp = temp.sample(25)
            temp['Sampling'] = 'Selected'
            alldata.append(temp)
        alldata = pd.concat(alldata)
        alldata.to_csv('all_units.csv')
        alldata[alldata.Sampling == 'Selected'].to_csv('selected_units.csv')
    
    return alldata

if __name__ == '__main__':
    
    data = Get_Data()
    Plot_Data(data[data.Sampling == 'All data'],'')
    # Plot_Data(data,'')