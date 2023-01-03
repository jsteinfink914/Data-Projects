#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:50:55 2022

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
# CREATE A BAR PLOT THAT SHOWS THE AVERAGE VALUE OF EACH FEATURE
#
# IN INTERACTIVE PLOT, WILL SHOW THE AVERAGE OVER THE SELECTED TIME
# 
# -------------------------------------------------------------------------- #

# The categories are:
    # 1. VALENCE
    # 2. ENERGY
    # 3. MODE
    # 4. DANCEABILITY
    # 5. KEY
    # 6. TEMPO
    
# select the columns you want 
complete_df_reduced = complete_df[['end_time', 'artist_name', 'track_name',
                                   'id'] + measures]

# may have to pivot the complete_df in order for this to work 
complete_melt = pd.melt(complete_df_reduced, 
                     id_vars = ['id', 'track_name', 'artist_name', 'end_time'],
                     var_name = 'feature')

# have to turn the values into percentages so they can be relative
# Maximum values for the categories are:
    # 1. 1
    # 2. 1
    # 3. 1
    # 4. 1
    # 5. 11
    # 6. 220
    

complete_melt['value'] = np.where(complete_melt['feature'] == 'key',
                                 complete_melt['value']/11,
                                 complete_melt['value'])

complete_melt['value'] = np.where(complete_melt['feature'] == 'tempo',
                                 complete_melt['value']/220,
                                 complete_melt['value'])

# complete_melt['value'] = complete_melt['value'] * 100 # for percentages

# now need to grouby this whole thing 
# make sure to keep the names/ids to identify
averages = (complete_melt.groupby(['feature'])
                  .mean().round(2)
                  .reset_index())                  
                    # [['id','end_time']])
                  # .rename(columns = {'end_time': 'average'})
                  # .sort_values('count', ascending = False)
                  # .reset_index(drop = True)
                  # )
   
# https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features                   
averages['description'] = [
    'Danceability describes how suitable a track is for dancing based on a<br>combination of musical elements including tempo, rhythm stability, beat strength,<br>and overall regularity.<br>A value of 0.0 is least danceable and 1.0 is most danceable.',
    'Energy represents a perceptual measure of intensity and activity.<br>Typically, energetic tracks feel fast, loud, and noisy<br>A value of 0.0 is least energetic and 1.0 is most energetic.',
    'The key the track is in.<br>Integers map to pitches using standard Pitch Class notation.<br>0 = C, 1 = C♯/D♭, 2 = D, and so on until 11.<br>If no key was detected, the value is -1.',
    'Mode indicates the modality (major or minor) of a track,<br>the type of scale from which its melodic content is derived.<br>Major chords are associated with happier songs,<br>while minor chords appear in more subdued songs.<br>Major is represented by 1 and minor is 0.',
    'The overall estimated tempo of a track in beats per minute (BPM).<br>The max tempo in this dataset is 220.',
    'A measure describing the musical positiveness conveyed by a track.<br>Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric),<br>while tracks with low valence sound more negative (e.g. sad, depressed, angry).<br>Valence values range from 0.0 to 1.0.'
    ]
              
# couldn't figure out the gradient
bar_colors = ['#DE3163', '#FB5607', '#028090', '#05204A', '#C70039', '#FFBF00']

"""
bar_colors = [(0,'#DE3163'), 
              (0,'#FB5607'), 
              (0,'#028090'), 
              (0,'#05204A'), 
              (0,'#C70039'), 
              (0,'#FFBF00')]
"""

percentages = px.bar(averages, 
                x='feature', 
                y='value',
                # text_auto = '0.2s',
                #color_continuous_scale = bar_colors,
                title = 'Average Values of Track Features Across All Streaming History',
                labels = dict(feature = 'Features', 
                          value = 'Percent (%)'),
                text_auto = True,
                hover_name = 'feature',
                hover_data = {
                    'feature': False,
                    'value': False,
                    'description': True # looks very odd
                    })

percentages.update_xaxes(
                         tickfont = dict(size = 16),
                         title_font=dict(size=20, color='#05204A'))

percentages.update_yaxes(
                         tickfont = dict(size = 16),
                         title_font=dict(size=20, color='#05204A'))

percentages.update_traces(textfont_size=20, # for labels
                          textangle=0, 
                          textposition="outside", 
                          cliponaxis=False
                          )


percentages.update_traces(marker_color=bar_colors) # for borders
                  # marker_line_color='#05204A',
                  # marker_line_width=2, 
                  # opacity=1)


percentages.update_layout(plot_bgcolor='#FFFFFF',
                          yaxis_range = [0, 1],
                          font_family = 'Optima',
                          font_color = '#05204A',
                          yaxis_tickformat = ',.0%',
                           title={
                               'y':0.95,
                               'x':0.5,
                               'font_size': 30,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           hoverlabel=dict(font_size=16,
                                           font_family="Optima"))

pio.write_html(percentages, file="fig-5b.html", auto_open=True)

















