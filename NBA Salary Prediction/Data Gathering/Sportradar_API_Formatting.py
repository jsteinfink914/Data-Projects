# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 10:43:47 2021

@author: jstei
"""
##Importing necessary libraries 
import pandas as pd
import json

# =============================================================================
# Opening schedule file to access each of the unique team_ids
# =============================================================================
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

# =============================================================================
# Opening each individual json file created in Sportradar_API converting it to a dataframe, and
# concatenating them together to make one large dataframe.
# =============================================================================

#list of years used
years=[2020,2019,2018,2017,2016,2015,2014,2013]
# =============================================================================
# Taking one of the files and isolating the different key values used in the files.
# This will be helpful below as some of the keys have values that themselves are dictionaries which must be converted.
# =============================================================================
filename=str(years[0])+'PlayerStats'+str(unique_team_ids[0])+'.json'
with open(filename,'r') as file:
    json_data=json.load(file)
json_data_keys=[i for i in json_data.keys()]
season_keys=[i for i in json_data['season'].keys()]
players_keys=[i for i in json_data['players'][0].keys()]
players_total_keys=[i for i in json_data['players'][0]['total'].keys()]
players_average_keys=[i for i in json_data['players'][0]['average'].keys()]

# =============================================================================
# This section loops through each individual file and creates the master dataframe
# =============================================================================
##Empty list to append each individual dataframe to
datasets=[]
##Nested for loops to access each file
for year in years:
    for i in range(len(unique_team_ids)):
        filename= str(year)+'PlayerStats'+str(unique_team_ids[i])+'.json'
        with open(filename,'r') as data_file:
                json_data=json.load(data_file)  
        #Creating a dataframe from the json file with a row being a singular player on the team
        StatsDF=pd.DataFrame.from_dict(json_data['players'])
        ##This is why the keys were isolated earlier
        ##The "total" and "average" columns return values that are dictionaries (each cell contains a host of key value pairs)
        ##To correct this, the pandas.Series() method is used to convert the dictionary into a set of key value pairs where the keys are columns and the values are the row values
        StatsDF[players_total_keys]=StatsDF.total.apply(pd.Series)
        StatsDF[players_average_keys]=StatsDF.average.apply(pd.Series)
        ##Dropping the 'total' and 'average' columns
        StatsDF=StatsDF.drop(['total','average'],axis=1)
        ##This for loop is used to create extra rows with valuable information from the json file
        ##Rows were created for year, team_id, team market, and team name 
        for i in range(len(StatsDF)):
            StatsDF.loc[i,'year']=int(str(year))
            StatsDF.loc[i,'team_id']=json_data['id']
            StatsDF.loc[i,'market']=json_data['market']
            StatsDF.loc[i,'name']=json_data['name']
        ##Collecting colum names and reordering them to put the year and team name as the first columns instead of the last
        cols=StatsDF.columns.tolist()
        cols=cols[-4:]+cols[:-4]
        StatsDF=StatsDF[cols]
        datasets.append(StatsDF)

##Concatenating the individual dataframes together and writing to a csv file
StatsDF_2020=pd.concat(datasets,axis=0)
StatsDF_2020.to_csv("2013-2020_NBA_Stats.csv",index=None)
