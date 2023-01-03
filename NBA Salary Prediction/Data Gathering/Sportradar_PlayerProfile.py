# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 12:25:33 2021

@author: jstei
"""
#Importing necessary libraries
import http.client
import json
import pandas as pd
import time
# =============================================================================
# Now to gather qualitative information on each player, access to the "Player Profile"
# Sportradar endpoint is needed. To do so, player_ids must be isolated from the dataframe
# =============================================================================
filename='CombinedDF_Salary.csv'
DF=pd.read_csv(filename)
##Isolating unique player ids
unique_player_ids=DF.id.unique()
unique_player_ids=[i for i in unique_player_ids]

# =============================================================================
# Using the unique_player_ids, a series of API calls will be made to get the 
# player profile for each playerand place the results into a file denoted by 
# the player id
# =============================================================================
for i in range(len(unique_player_ids[:600])):
    conn = http.client.HTTPSConnection("api.sportradar.us")
    conn.request("GET", "/nba/trial/v7/en/players/"+
                 str(unique_player_ids[i])+
                 "/profile.json?api_key=matv5tcqvhsjx9kfut3ay4mm")
    res = conn.getresponse()
    data = res.read()
    json_txt=json.loads(data)
    with open("PLayerProfiles/PlayerProfile"+
              str(unique_player_ids[i])+".json",'w') as file:
        json.dump(json_txt,file)
    time.sleep(1)

# =============================================================================
# The API call is split into 2 different sections because the original free trial was limited to 1000 API calls
# To overcome this limit, another free trial was created to use another key
# Despite this difference, everything else is the same
# =============================================================================
for i in range(len(unique_player_ids[600:])):
    conn = http.client.HTTPSConnection("api.sportradar.us")
    conn.request("GET", "/nba/trial/v7/en/players/"+
                 str(unique_player_ids[(1161+i)])+
                 "/profile.json?api_key=8nctux22m3pvkmj4fqv6tfn4")
    res = conn.getresponse()
    data = res.read()
    json_txt=json.loads(data)
    with open("PlayerProfiles/PlayerProfile"+
              str(unique_player_ids[(1161+i)])+
              ".json",'w') as file:
        json.dump(json_txt,file)
    time.sleep(1)

# =============================================================================
# Applying the nested for loop technique again to match up the correct player file 
# with the correct cell in the parent dataframe and transfer
# height, weight, and experience data to the parent dataframe.
# Instead of using a 'combined' column with the player name and season, since the
# file is demarcated by a player_id and the parent dataframe holds an id column
# =============================================================================
for a in range(len(unique_player_ids)):
    with open("PlayerProfiles/PlayerProfile"
              +str(unique_player_ids[a])+".json",'r') as file:
        json_data=json.load(file)
    ##accessing height, weight, birthdate, and experience directly from the json file
    height=json_data['height']
    weight=json_data['weight']
    birthdate=json_data['birthdate']
    ##try and except sturcture is used because player's with 
    ##0 years of experience have no experience attribute in the json data
    try:
        experience=json_data['experience']
    except:
        experience=0
    for i in range(len(DF)):
        playerid=DF.loc[i,'id']
        if playerid==unique_player_ids[a]:
            DF.loc[i,'height']=height
            DF.loc[i,'weight']=weight
            DF.loc[i,'experience']=experience
            DF.loc[i,'birthdate']=birthdate
# =============================================================================
# Important Note: this structure of assigning experience results in the same experience
# value for a player regardless of the year. For instance, a player with 12 years of experience
# as of 2020 also has 12 in the experience column for 2019. This issue will be dealt with later.
# =============================================================================
##Overwriting the same csv file
DF.to_csv('CombinedDF_Salary.csv',index=None)



                
                    


