#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:10:58 2022

@author: ninathomas
"""

# -------------------------------------------------------------------------- #
#
# CREATE CSV FILE OF STREAMING HISTORY 
#
# -------------------------------------------------------------------------- #

import json
import pandas as pd
import requests 
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import time
import spotipy

pd.set_option('max_columns', None)

# -------------------------------------------------------------------------- #
#
# SET UP ACCESS TO SPOTIFY API
# 1. Set up a Spotify Developer account
# 2. Go to "Dashboard"
# 3. Create a new application
# 4. You will find client_id and client_secret by clicking on your application.
# 5. For the redirect_uri, use any URL
#
# -------------------------------------------------------------------------- #

spotify_url = 'https://api.spotify.com/v1'

client_id = '' 

client_secret = ''

redirect_uri = ''

scope = 'user-library-read'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = client_id,
                                               client_secret = client_secret,
                                               redirect_uri = redirect_uri,
                                               scope=scope))

# -------------------------------------------------------------------------- #
#
# DEFINE FUNCTIONS
#
# -------------------------------------------------------------------------- #
def get_id(row):
   
    # 'artist:' + row['artist_name'] + 
    track = sp.search(q = 'track:' + row['track_name'], 
                         type='track')
    try:
        track_id = track['tracks']['items'][0]['id'] # there are 7 elements per track_id['tracks']
        return track_id
    except:
        print(row['track_name'], 'is having issues...')
        return 'NA'
        
# -------------------------------------------------------------------------- #
#
# READ IN JSON FILES TO DICTIONARIES
#
# -------------------------------------------------------------------------- #

json_0 = 'StreamingHistory0.json'
json_1 = 'StreamingHistory1.json'
json_2 = 'StreamingHistory2.json'

list_jsons = [json_0, json_1, json_2]

# -------------------------------------------------------------------------- #
#
# CREATE COMPLETE DATAFRAME OF STREAMING DATA 
#
# -------------------------------------------------------------------------- #

streaming_history = pd.DataFrame()

for json_num in list_jsons:
    # read json to dict
    with open(json_num, 'r') as json_file:
        history_dict = json.load(json_file) 
        
    # convert from dict to df
    history_df = pd.DataFrame.from_dict(history_dict)
    
    streaming_history = pd.concat([streaming_history, history_df], axis = 0)

streaming_history = streaming_history.rename(columns = {'endTime': 'end_time', 
                                                        'artistName': 'artist_name',
                                                        'trackName': 'track_name',
                                                        'msPlayed': 'ms_played',
                                                        })

# -------------------------------------------------------------------------- #
#
# ADD COLUMN WITH ALL THE SONG IDS
#
# obtain track name from the id
# short_stream = streaming_history.iloc[0:20]
# short_stream['id'] = short_stream.apply(get_id, axis = 1) 
# sp.track(short_stream.iloc[0]['id'])['name']
#
# -------------------------------------------------------------------------- #

streaming_history['id'] = streaming_history.apply(get_id, axis = 1) # add the id

is_null = streaming_history[streaming_history['id'].isnull()] # obtain songs without ids
 
streaming_history_clean = streaming_history[~streaming_history['id'].isnull()] # remove songs without ids
    
# -------------------------------------------------------------------------- #
#
# CREATE CSV FILES
#
# -------------------------------------------------------------------------- #
is_null.to_csv('is_null.csv', index = False) # missing songs
streaming_history_clean.to_csv('streaming_history.csv', index = False) # write to csv


          

























