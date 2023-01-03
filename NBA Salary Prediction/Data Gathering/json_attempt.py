# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 10:43:47 2021

@author: jstei
"""

import pandas as pd
import json
#from itertools import chain
filename= '2020schedule.json'
with open(filename,'r') as data_file:
    json_data=json.load(data_file)
TeamID=pd.DataFrame.from_dict(json_data['games'])
away_keys=[i for i in json_data['games'][0]['away'].keys()]
home_keys=[i for i in json_data['games'][0]['home'].keys()]
TeamID[away_keys]=TeamID.away.apply(pd.Series)
TeamID[home_keys]=TeamID.home.apply(pd.Series)
unique_team_ids=TeamID.id.unique()
unique_team_ids=[i for i in unique_team_ids]
unique_team_ids=unique_team_ids[:30]

filename='PlayerStats.json'
with open(filename,'r') as file:
    json_data=json.load(file)
json_data_keys=[i for i in json_data.keys()]
season_keys=[i for i in json_data['season'].keys()]
players_keys=[i for i in json_data['players'][0].keys()]
players_total_keys=[i for i in json_data['players'][0]['total'].keys()]
players_average_keys=[i for i in json_data['players'][0]['average'].keys()]

datasets=[]
for i in range(len(unique_team_ids)):
    filename= '2020PlayerStats'+str(unique_team_ids[i])+'.json'
    with open(filename,'r') as data_file:
            json_data=json.load(data_file)    
    StatsDF=pd.DataFrame.from_dict(json_data['players'])
    StatsDF[players_total_keys]=StatsDF.total.apply(pd.Series)
    StatsDF[players_average_keys]=StatsDF.average.apply(pd.Series)
    StatsDF=StatsDF.drop(['total','average'],axis=1)
    for i in range(len(StatsDF)):
        StatsDF.loc[i,'year']=int(2020)
        StatsDF.loc[i,'team_id']=json_data['id']
        StatsDF.loc[i,'market']=json_data['market']
        StatsDF.loc[i,'name']=json_data['name']
    cols=StatsDF.columns.tolist()
    cols=cols[-4:]+cols[:-4]
    StatsDF=StatsDF[cols]
    datasets.append(StatsDF)

StatsDF_2020=pd.concat(datasets,axis=0)
StatsDF_2020.to_csv("2020_NBA_Stats.csv",index=None)
