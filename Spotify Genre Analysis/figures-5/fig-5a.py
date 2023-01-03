#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:24:04 2022

@author: ninathomas
"""
import json
import pandas as pd
import requests 
import numpy as np
from datetime import datetime

import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import time
# import Image
import spotipy

# -------------------------------------------------------------------------- #
#
# ALLOW FOR INTERACTIVE GRAPHICS
# 
# -------------------------------------------------------------------------- #

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

from statsmodels.tsa.seasonal import seasonal_decompose

# https://dash.plotly.com/interactive-graphing
from dash import Dash, html, dcc, Input, Output 

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
complete_df = pd.read_csv('combined_df.csv')
measures = ['valence', 'energy', 'mode', 'danceability', 'key', 'tempo']

# -------------------------------------------------------------------------- #
#
# THIS BAR PLOT WILL LOOK SOLELY AT THE FREQUENCY OF SONGS IN A SELECTED
# TIME SPAN/PLAYLIST
#
# IT WILL BE SHOWN IN CONJUNCTION WITH THE FEATURE_RANKINGS BAR GRAPH
# SO THAT THE STATISTICS WILL BE VISIBLE
#
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
#
# CODE FOR CREATING A FREQUENCY BAR PLOT
#
# -------------------------------------------------------------------------- #

# obtain the frequencies for each song id
counts_listened = (complete_df.groupby(['id'])
                  .count()
                  .reset_index()
                  [['id','end_time']] # select two columns
                  .rename(columns = {'end_time': 'count'}) # rename to count
                  .sort_values('count', ascending = False) # rank from greatest to least
                  .reset_index(drop = True))

# df with counts now, merge based on song id
complete_df_count = complete_df.merge(counts_listened, how = 'left',
                                on = 'id')

# remove the rows with not enough data
complete_df_count_clean = complete_df_count[complete_df_count['count'] > 5]

# use groupby so that each song appears only once
grouped_complete_count = (complete_df_count_clean
                          .groupby(['track_name', 'artist_name', 'id']).mean('count')
                          .sort_values('count', ascending = False)
                          .reset_index())

# remove songs that have duplicate track names
grouped_clean_count = grouped_complete_count.drop_duplicates(subset='track_name', keep=False)
                                                       
# one song doesn't seem to belong... this is because the query in the initial data would
# occassionally lead to the incorrect result
# for the most time, it's correct
counts = px.bar(grouped_clean_count.iloc[0:15], # top 15 songs
                x='track_name', 
                y='count',
                title = 'Top 15 Songs In Selected Period',
                hover_name = 'track_name', # title is the song name
                # hover_data =  ['artist_name'] + measures,
                color = 'count',
                text_auto = True,
                labels = dict(track_name = 'Track', 
                          count = '# of Times Played'),
                color_continuous_scale = 'emrld',
                hover_data = {
                    'track_name': False,
                    'count': False,
                    'artist_name': True,
                    'danceability': True,
                    'energy': True,
                    'key': True,
                    'mode': True,
                    'tempo': True,
                    'valence': True}
                ) # includes the feature values


counts.update_traces(textfont_size=16, # for labels
                    textangle=0, 
                    textposition="outside", 
                    cliponaxis=False)

counts.update_xaxes(showticklabels = False,
                    tickfont = dict(size = 16),
                    title_font=dict(size=20, color='#05204A'),
                    tickangle = 35)

counts.update_yaxes(
                    tickfont = dict(size = 16),
                    title_font=dict(size=20, color='#05204A'))

# here's where we can have a gradient of colors
counts.update_layout(
    plot_bgcolor='#FFFFFF',
    yaxis_range = [0, max(grouped_clean_count['count'])],
    title_font_size = 30,
    width = 800,
    height = 650,
    font_color = '#05204A',
    font_family = 'Optima',
    hoverlabel=dict(
        font_size=16,
        font_family = "Optima"),
    title={
        'y':0.95,
        'x':0.45,
        'font_size': 30,
        'xanchor': 'center',
        'yanchor': 'top'})

pio.write_html(counts, file="fig-5a.html", auto_open=True)







