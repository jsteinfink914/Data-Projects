# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:41:14 2021

@author: jstei
"""
# =============================================================================
# This code creates a dataframe for all duplicate entries
# (entries that are for the same player in a given season due to 
# team change) made in the raw data by isolating and averaging/summing 
# the necessary information to form a complete row storing 
# the players yearly stats.
# =============================================================================
#Importing necessary libraries 
import pandas as pd

##Read in the raw data
DF=pd.read_csv('RawDF.csv')

##Dropping 2 index rows duplicated in the past, 
##'jersey_number' and 'reference' are meaningless numeric columns
labels=['Unnamed: 0','Unnamed: 0.1','jersey_number','reference']
DF=DF.drop(labels, axis=1)

##Year should be retained outright without any mathematical operations
DF.year=DF.year.astype('object')

##Initializing a new column called team_changes to track player movement
DF['team_changes']=0

##Initializing 2 empty lists which will store duplicate 
##players' names and their index values
duplicates=[]
drop_index_values=[]

#For loop used to isolate duplicated entries and their indexes
for a in range(len(DF)):
    name=DF.loc[a,'combined']
    values=DF[DF.combined==name].index.values
    values=[i for i in values]
    ##If loop used to identify duplicates
    if len(values)>1:
        duplicates.append(name)
        drop_index_values.append(values)

##Isolating unique names and printing to console
duplicates=pd.unique(duplicates)
print("There are "+str(len(duplicates))+" players who changed teams in the same season")
##Isolating unique duplicate indexes and printing to console
drop_index_values=[i for sublist in drop_index_values for i in sublist]
drop_index_values=pd.unique(drop_index_values)
print("There are "+str(len(drop_index_values))+" duplicate entries in the data.")
print("This accounts for "+str(round((len(drop_index_values)/len(DF))*100,2))+"% of the "+str(len(DF))+" rows in the raw data.")

##Collecting column keys that will be looped through 
##to collect the data 
keys=[i for i in DF.columns]
columns=dict.fromkeys(keys)

##Empty list that will store the individual dataframes 
##for each duplicate
dataframes=[]

##Identifying columns that are 'total' columns and have
##to be summed instead of averaged
sum_columns=['games_played','games_started','offensive_rebounds','defensive_rebounds','tech_fouls','ejections','foulouts','double_doubles','triple_doubles']

##Lopping through each duplicate
for i in duplicates:
    ##Initializing empty list for the duplicate's data
    data=[]
    ##collecting index values
    values=DF[DF.combined==i].index.values
    values=[i for i in values]
    ##lopping through each column
    for column in columns:
        ##Empty list to process the collection of data points
        point=[]
        ##Looping through each index value for the duplicate
        for value in values:
            point.append(DF.loc[value,column])
        ##Isolating games_played column as they serve as the
        ##weights to average per game stats
        if column=="games_played":
            weights=[]
            ##lopping through each point and calculating 
            ##weights
            for i in range(len(point)):
                weights.append(point[i]/sum(point))
            data.append(sum(point))
# =============================================================================
# Duplicates were created as a result of team changes
# in the middle of the season. The number of duplicate index
# values for a given duplicate minus 1 represents the 
# number of team changes 
# =============================================================================
        elif column=='team_changes':
            data.append(len(values)-1)
        ##Identifying columns that need to be summed
        elif any(i==column for i in sum_columns):
            data.append(sum(point))
        ##All other columns are either strings or need to be averaged
        else:
            ##If the column is a string column, use the entrance 
            ##from one of the values (does not matter which)
            if DF[column].dtype=='O':
                data.append(DF.loc[values[0],column])
            else:
                ##Performing weighted averages
                product=[]
                ##Comnbining weights list with their respective
                ##totals and computing a weighted average
                for num1,num2 in zip(weights,point):
                    product.append(num1*num2)
                data.append(sum(product))
    ##Creating a dictionary with column names as keys and 
    ##the averages/sums/objects as values
    result=dict(zip(columns,data))
    ##Creating a dataframe from a dictionary for each duplicate
    individualDF=pd.DataFrame(result,index=[0])
    ##Appending them to the dataframes list
    dataframes.append(individualDF)

##Creating a duplicate dataframe
DuplicateDF=pd.concat(dataframes,axis=0)

##Dropping duplicate entries from the dataframe
DF=DF.drop(drop_index_values,axis=0)

##Joining the duplicate dataframe to the raw dataframe
Duplicate_Free_DF=pd.concat([DF,DuplicateDF],axis=0)

##Converting the year column back to integer
DF.year=DF.year.astype('int64')
Duplicate_Free_DF.year=Duplicate_Free_DF.year.astype('int64')

##Sorting the dataframe by year
Duplicate_Free_DF=Duplicate_Free_DF.sort_values(by=['year'],ascending=False)

##Writing both datasets to csv files
#DuplicateDF.to_csv('DuplicateDF.csv',index=None)
#Duplicate_Free_DF.to_csv('Duplicate_Free_DF.csv',index=None)

##Printing new # of rows and unique players to the console
print("There are "+str(len(Duplicate_Free_DF))+" rows in the new dataframe,")
print("and "+str(len(pd.unique(Duplicate_Free_DF.id)))+" unique players.")    







