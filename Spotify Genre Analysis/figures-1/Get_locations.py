# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:43:31 2022

@author: jstei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:14:59 2022

@author: jstei
"""

# -------------------------------------------------------------------------- #
#
# FIND ARTIST AND SONG IDS
#
# -------------------------------------------------------------------------- #

import json
import pandas as pd
import requests 
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import time
# import Image
import spotipy

pd.set_option('max_columns', None)

# -------------------------------------------------------------------------- #
#
# SET UP ACCESS TO SPOTIFY API
# 
# -------------------------------------------------------------------------- #

spotify_url = 'https://api.spotify.com/v1'

client_id = '8af7373ce2f94ec8841b199abd81321c'

client_secret = 'fb480348d1d44e8994e47234254a30a6'

redirect_uri = 'https://cathomas.georgetown.domains/ANLY560/AboutMe.html'

# client_credentials_manager = SpotifyClientCredentials(client_id = client_id, 
                                                      # client_secret = client_secret)
scope = 'user-library-read'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = client_id,
                                               client_secret = client_secret,
                                               redirect_uri = redirect_uri,
                                               scope=scope))


Songs = pd.read_csv('TopSongs.csv')
DF = pd.DataFrame(columns=Songs.columns.tolist())

for i in range(len(Songs)):
    song= Songs.loc[i,'title']
    artist= Songs.loc[i,'artist']
    # returns list of 7 songs named 'Lost'
    songs = sp.search(q = 'track:' + song, type='track')
    artists = sp.search(q='artist:' + artist, type='artist') 
    # to find the artist id:
    artist_id = artists['artists']['items'][0]['id'] 
    # loop through song search results and compare artist id for each song 
    for a in songs['tracks']['items']:
        song_artist_id = a['artists'][0]['id']
        if song_artist_id == artist_id:
            found_song_name = a['name']
            markets = a['available_markets']
            for z in range(len(markets)):
                 row = Songs.loc[i]
                 row['Country'] = markets[z]
                 row['uri'] = a['uri']
                 DF = DF.append(row,ignore_index=True)
            break
            


DF_counts = DF.groupby(by=['Country']).size()
DF_counts = DF_counts.to_frame().reset_index()
DF_counts = DF_counts.rename(columns = {0:'Counts'})           
DF_counts['Percentage'] = DF_counts.Counts/max(DF_counts.Counts)
DF.to_csv('Map.csv',index=False)







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    