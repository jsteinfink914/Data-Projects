# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 20:22:46 2021

@author: jstei
"""
# =============================================================================
# This code will take the duplicate free data and 
# edit the experience and team changes columns.
# Currently, the experience column is static for players 
# in that a players experience value is based on their experience
# as of their last playing year, and that entry is the same for all years
# =============================================================================
import pandas as pd
import datetime as dt
DF=pd.read_csv('Duplicate_Free_DF.csv')

##Now that all players only have one entry per year 
##To track experience correctly, all instances of the player
##must be collected and the experience subtracted by 1 for each
##instance of the player outside of their most recent
##playing year.

##Intializing empty list to ensure that the operation is
##only done once
Completed_names=[]

##Using for loop to collect player indexes
for i in range(len(DF)):
    name=DF.loc[i,'full_name']
    ##Skipping over names already done
    if any(i==name for i in Completed_names):
        continue
    else:
        ##Collecting index values
        values=DF[DF.full_name==name].index.values
        values=[i for i in values]
        ##looping through each value to edit the experience value
        for i in range(len(values)):
            DF.loc[values[i],'experience']=DF.loc[values[i],'experience']-i
            ##Some players have negative experience values if they missed
            ##years due to injury, to rectify this, set the minimum at 0
            if DF.loc[values[i],'experience']<0:
                DF.loc[values[i],'experience']=0
    Completed_names.append(name)


# =============================================================================
# For team changes, those who had multiple entries already have their
# team changes accounted for. For the rest (and the additional changes
# for those who moved between seasons), to find the number
# of teams they have been on, collect the indexes for the player
# create a vector of their teams and additional team changes
# equals the amount of unique teams
# 
# IMPORTANT NOTE: to track team changes appropriately through
# time this operation has to be performed with the dataframe
# in chronological order
# =============================================================================
DF=DF.sort_values(by=['year'])

for i in range(len(DF)):
    name=DF.loc[i,'full_name']
    team_vector=[]
    values=DF[DF.full_name==name].index.values
    values=[i for i in values]
    for i in range(len(values)):
        team_vector.append(DF.loc[values[i],'market'])
        DF.loc[values[i],'team_changes']=len(pd.unique(team_vector))-1
        
DF=DF.sort_values(by=['year'],ascending=False)

# =============================================================================
# To create an Age variable, the birthdate must be subtracted from the season 
# year taking into account the fact that the NBA season starts in October
# =============================================================================
for i in range(len(DF)):
    birthyear=int(DF.loc[i,'birthdate'][:4])
    birthmonth=DF.loc[i,'birthdate'][5:7]
    if birthmonth[:1]=='0':
        birthmonth=int(DF.loc[i,'birthdate'][6:7])
    else:
        birthmonth=int(birthmonth)
    birthday=DF.loc[i,'birthdate'][8:10]
    if birthday[:1]=='0':
        birthday=int(DF.loc[i,'birthdate'][9:10])
    else:
        birthday=int(birthday)
    birthdate=dt.date(birthyear,birthmonth,birthday)
    season=int(DF.loc[i,'year'])
    end_date=dt.date(season,10,1)
    time_difference=end_date-birthdate
    age=time_difference.days
    age=age//365
    DF.loc[i,'age']=age
    

DF.to_csv('DF_with_experience.csv',index=None)


