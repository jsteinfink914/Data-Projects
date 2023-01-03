# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:27:27 2021

@author: jstei
"""
import pandas as pd

DF=pd.read_csv('DF_SalaryCap.csv')
All_Nba=pd.read_csv('All_NBA.csv',encoding='cp1252')
All_Nba.yearSeason=All_Nba.yearSeason-1
for i in range(len(All_Nba)):
    All_Nba.loc[i,'combined']=All_Nba.loc[i,'namePlayer']+str(All_Nba.loc[i,'yearSeason'])

All_Nba=All_Nba.drop([i for i in range(137,885)])

for a in range(len(All_Nba)):
    name=All_Nba.loc[a,'combined']
    for i in range(len(DF)):
        name2=DF.loc[i,'combined']
        if name2==name:
            DF.loc[i,'All_Nba']=All_Nba.loc[a,'isAllNBA']
            DF.loc[i,'All_Nba_team']=All_Nba.loc[a,'numberAllNBATeam']

DF.to_csv('RawDF.csv',index=None)
