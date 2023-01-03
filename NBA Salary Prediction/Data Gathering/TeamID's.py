# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:26:57 2021

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


filename= '2019schedule.json'
with open(filename,'r') as data_file:
    json_data=json.load(data_file)
TeamID2=pd.DataFrame.from_dict(json_data['games'])
away_keys=[i for i in json_data['games'][0]['away'].keys()]
home_keys=[i for i in json_data['games'][0]['home'].keys()]
TeamID2[away_keys]=TeamID2.away.apply(pd.Series)
TeamID2[home_keys]=TeamID2.home.apply(pd.Series)
Aunique_team_ids=TeamID2.id.unique()
Aunique_team_ids=[i for i in Aunique_team_ids]
Aunique_team_ids=Aunique_team_ids[:30]

count=0
