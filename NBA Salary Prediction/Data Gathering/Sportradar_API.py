# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:51:01 2021

@author: jstei
"""
##Importing necessary libraries
import http.client
import json
import pandas as pd
import time
# =============================================================================
# Before using the Sportradar API to scrape player stats data from the endpoint 
# "Seasonal Statistics" which returns season statistics for a given team (and all players on that team for the year)
# team_ids must first be collected. These are the same for all years, so 2020 is used 
# to access the "Schedule" endpoint which hosts this information.
# =============================================================================
##Collecting team_ids
conn = http.client.HTTPSConnection("api.sportradar.us")
##Making the API request
conn.request("GET", "/nba/trial/v7/en/games/2020/REG/schedule.json?api_key=matv5tcqvhsjx9kfut3ay4mm")
##Collecting response
res = conn.getresponse()
##Converting the response  into readable format
data = res.read()
##Converting the response into json
schedule_txt=json.loads(data)
##Placing the json formatted response into "2020schedule.json"
with open('2020schedule.json','w') as file:
    json.dump(schedule_txt,file)

# =============================================================================
# Now the json formatted response is housed in a file which must be used to gather the team_ids
# =============================================================================
filename= '2020schedule.json'
with open(filename,'r') as data_file:
    json_data=json.load(data_file)
##Converting the "games" dictionaries in the file into a pandas dataframe
TeamID=pd.DataFrame.from_dict(json_data['games'])
##Using list comprehension to access the various keys housed in the "game" tag which are included under the sub tags of "home" and "away"
away_keys=[i for i in json_data['games'][0]['away'].keys()]
home_keys=[i for i in json_data['games'][0]['home'].keys()]
# =============================================================================
# Important note: at this point the pandas dataframe has normal columns and rows except for the
# home and away columns which host dictionaries in each cell.
# The .apply(pandas.series) method converts these dictionaries into a normal dataframe with the keys as columns and values as the cell values.
# =============================================================================
TeamID[away_keys]=TeamID.away.apply(pd.Series)
TeamID[home_keys]=TeamID.home.apply(pd.Series)
## Now the dataframe looks entirely normal. Using pandas .unique() method, it is straightforward to collect the unique values in the "id" column (which represent team_ids)
unique_team_ids=TeamID.id.unique()
unique_team_ids=[i for i in unique_team_ids]
##The last unique team_id was for the all-star game so this one is neglected as it is not a real NBA team
unique_team_ids=unique_team_ids[:30]

# =============================================================================
# Now that all the team_ids are gathered, 2 nested for loops are used to access
# each year, and then each team within that year.The results for each team for 
# each season are converted into json and dumped into a file which will then be 
# processed later. 
# =============================================================================
##Collecting data for each team from 2012-2020
years=[2020,2019,2018,2017,2016,2015,2014,2013,2012]
for year in years:
    for i in range(len(unique_team_ids)):
        conn = http.client.HTTPSConnection("api.sportradar.us")
        conn.request("GET", "/nba/trial/v7/en/seasons/"+str(year)+
                     "/REG/teams/"+str(unique_team_ids[i])+
                     "/statistics.json?api_key=matv5tcqvhsjx9kfut3ay4mm")
        res = conn.getresponse()
        data = res.read()
        json_txt=json.loads(data)
        with open(str(year)+"PlayerStats"+str(unique_team_ids[i])+".json",'w') as file:
            json.dump(json_txt,file)
        ##The time.sleep(1) piece is necessary because the API call is rate-limited to 1 per second
        time.sleep(1)

    

