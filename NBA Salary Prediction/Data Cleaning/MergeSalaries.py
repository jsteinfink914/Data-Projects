# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:24:08 2021

@author: jstei
"""
# =============================================================================
# This code appends NBA salaries from 2012-2020 received from 2 kaggle dataframes
# sourced from: https://www.kaggle.com/josejatem/nba-salaries-20032019  and 
# https://www.kaggle.com/junfenglim/nba-player-salaries-201920
# =============================================================================
#Importing pandas
import pandas as pd

#Importing parent dataframe
StatsDF=pd.read_csv('CombinedDF.csv')
#Importing first salary dataframe with salaries from 2003-2018
Salaries_2003_2019=pd.read_csv('salaries 2003-2019.csv')
#Importing second salary dataframe with 2019 salaries
Salaries_2019=pd.read_csv('nba2019-2020salaries.csv')
# =============================================================================
# Editing the 2003-2018 salary dataframe to subtract one from the year
# This change was made because this dataframe has the year listed
# as the year when the season ended as opposed to when the season began
# which is how the parent dataframe is formatted.
# =============================================================================
Salaries_2003_2019.Season=(Salaries_2003_2019.Season-1)
#Performing the same operation as in the 2k dataframe
#'season' was formatted as '2019-2020' when only '2019' is needed
for i in range(len(Salaries_2019)):
    Salaries_2019.loc[i,'season']=Salaries_2019.loc[i,'season'][:4]

##Creating a 'combined' column for both datasets to link player name with the year
for i in range(len(Salaries_2003_2019)):
    Salaries_2003_2019.loc[i,'combined']=Salaries_2003_2019.loc[i,'Player']+str(Salaries_2003_2019.loc[i,'Season'])
for i in range(len(Salaries_2019)):
    Salaries_2019.loc[i,'combined']=Salaries_2019.loc[i,'player']+Salaries_2019.loc[i,'season']

# =============================================================================
# Running 2 sets of nested for loops (one for each salary dataset) to append salary values
# to the parent dataframe
# =============================================================================
for a in range(len(Salaries_2003_2019)):
    name=Salaries_2003_2019.loc[a,'combined']
    for i in range(len(StatsDF)):
        name2=StatsDF.loc[i,'combined']
        if name2==name:
            StatsDF.loc[i,'Salary']=Salaries_2003_2019.loc[a,'Salary']
            
for a in range(len(Salaries_2019)):
    name=Salaries_2019.loc[a,'combined']
    for i in range(len(StatsDF)):
        name2=StatsDF.loc[i,'combined']
        if name2==name:
            StatsDF.loc[i,'Salary']=Salaries_2019.loc[a,'salary']
##Writing the appended dataframe to a new csv file
StatsDF.to_csv('CombinedDF_Salary.csv')
            