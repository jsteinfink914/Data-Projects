# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:54:59 2021

@author: jstei
"""
# =============================================================================
# This code appends NBA2k rankings from 2014-2019 received from the 
# kaggle dataframe sourced from:'https://www.kaggle.com/willyiamyu/nba-2k-ratings-with-real-nba-stats'
# =============================================================================
##Importing pandas
import pandas as pd

##Reading the Parent dataframe in created from the sportradar api
StatsDF=pd.read_csv('2013-2020_NBA_Stats.csv')
##Reading in the downloaded kaggle dataset.
NBA2kDF=pd.read_csv('nba_rankings_2014-2020.csv')

##Editing the 'Season' column in the 2k dataset 
##Original entries appear as '2019-20' and only '2019' is needed
for i in range(len(NBA2kDF)):
    NBA2kDF.loc[i,'SEASON']=NBA2kDF.loc[i,'SEASON'][:4]
    
##Creating a 'combined' column in both dataframes that combines a players name
##and season for easier matching
for i in range(len(StatsDF)):
    StatsDF.loc[i,'combined']=StatsDF.loc[i,'full_name']+str(str(StatsDF.loc[i,'year'])[:4])
for i in range(len(NBA2kDF)):
    NBA2kDF.loc[i,'combined']=NBA2kDF.loc[i,'PLAYER']+str(NBA2kDF.loc[i,'SEASON'])

##Nested for loops used to match appropriate player-season combinations in 
##both dataframes to assign 2k ratings to the parent dataframe
for a in range(len(NBA2kDF)):
    name=NBA2kDF.loc[a,'combined']
    for i in range(len(StatsDF)):
        name2=StatsDF.loc[i,'combined']
        if name2==name:
            StatsDF.loc[i,'2K_ratings']=NBA2kDF.loc[a,'rankings']
            
##Writing the appended parent df to a new csv file
StatsDF.to_csv("CombinedDF.csv")
