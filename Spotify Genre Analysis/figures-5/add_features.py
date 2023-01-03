#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 01:05:31 2022

@author: ninathomas
"""

import json
import pandas as pd
import requests 
import numpy as np

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

streaming_history = pd.read_csv('streaming_history.csv')

def get_track_features(track_id):
    print(track_id)
    
    
    # THIS PART IS NOT WORKING!!
    # 
    # meta = sp.track(track_id) # meta data (having trouble calling)
    features = sp.audio_features(track_id)
    
    # some meta data is already in the dataset/not necessary
    # name = meta['name'] # song name
    # album = meta['album']['name'] # song album
    # artist = meta['album']['artists'][0]['name'] # artist name
    """
    release_date = meta['album']['release_date'] # release date
    length = meta['duration_ms'] # length 
    popularity = meta['popularity'] # song popularity
    images = meta['album']['images'][0]['url'] # album image
    preview = meta['preview_url'] # song preview
    """
    
    # include all features
    try:
    
        acousticness = features[0]['acousticness']
        danceability = features[0]['danceability'] # 
        energy = features[0]['energy'] #
        key = features[0]['key'] #  -1 = no key detected, 0 = C, 1 = C#/Dflat etc.
        instrumentalness = features[0]['instrumentalness']
        liveness = features[0]['liveness']
        loudness = features[0]['loudness'] # -60 to 0 decibels
        mode = features[0]['mode'] # 1 major, 0 minor
        speechiness = features[0]['speechiness'] 
        valence = features[0]['valence'] # higher valence is more positive
        tempo = features[0]['tempo'] # beats per minute
        time_signature = features[0]['time_signature']
        
        track = [#release_date,
                 #length,
                 #popularity,
                 #images,
                 #preview,
                 acousticness,
                 danceability,
                 energy,
                 key,
                 instrumentalness,
                 liveness,
                 loudness,
                 mode,
                 speechiness,
                 valence,
                 tempo,
                 time_signature]
        
        print(track)
        
        return track
    except:
        print(track_id, 'did not load properly.')
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# np.where(a[:,1]>5,a[:,0],0.1*a[:,0]*a[:,1]) this function may help 

tracks = [] # at 2907
for track_id in streaming_history['id']:
    track = get_track_features(track_id)
    tracks.append(track)
    
# print(streaming_history.iloc[0])
# track_id = streaming_history.iloc[0]
# track = get_track_features(track_id['id'])
# tracks.append(track)

# create dataset
all_features_df = pd.DataFrame(tracks, 
                  columns = [#'release_date',
                           #'length',
                           #'popularity',
                           #'images',
                           #'preview',
                           'acousticness',
                           'danceability',
                           'energy',
                           'key',
                           'instrumentalness',
                           'liveness',
                           'loudness',
                           'mode',
                           'speechiness',
                           'valence',
                           'tempo',
                           'time_signature'])

combined_df = streaming_history.merge(all_features_df, on = None, how = 'inner',
                                      left_index = True,
                                      right_index = True)

combined_df = combined_df[combined_df['time_signature'] != 0]
# so far up to 2021-03-29 14:03
# now up to 2021-07-12 04:33
# now up to 2022-02-16! 

# combined_df.to_csv('combined_df.csv', index = False) 
