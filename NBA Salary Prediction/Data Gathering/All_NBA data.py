# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:27:27 2021

@author: jstei
"""
##Importing pandas
import pandas as pd

##Reading in dataframes
DF=pd.read_csv('DF_SalaryCap.csv')
All_Nba=pd.read_csv('All_NBA.csv',encoding='cp1252')
##Subtracting 'yearSeason' column entries by 1 to reflect year at the beginning of the season
All_Nba.yearSeason=All_Nba.yearSeason-1
##Creating a combined column to pair player name with year
for i in range(len(All_Nba)):
    All_Nba.loc[i,'combined']=All_Nba.loc[i,'namePlayer']+str(All_Nba.loc[i,'yearSeason'])

##Dropping data for years pre 2012
All_Nba=All_Nba.drop([i for i in range(137,885)])

##Using nested for loop structure to match up the dataframes
for a in range(len(All_Nba)):
    name=All_Nba.loc[a,'combined']
    for i in range(len(DF)):
        name2=DF.loc[i,'combined']
        if name2==name:
            DF.loc[i,'All_Nba']=All_Nba.loc[a,'isAllNBA']
            DF.loc[i,'All_Nba_team']=All_Nba.loc[a,'numberAllNBATeam']

##Writing output to a final csv file refelcting the entire raw dataset
DF.to_csv('RawDF.csv',index=None)
