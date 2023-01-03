#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:56:18 2022

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

from ipywidgets import interactive, HBox, VBox

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
measures = ['danceability', 'energy', 'key', 'mode', 'tempo', 'valence']

# -------------------------------------------------------------------------- #
#
# NEED A WAY TO VISUALIZE THE SONGS WITH THE GREATEST/LEAST VALUES
#
# AND WHEN THEY WERE LISTENED TO 
#
# -------------------------------------------------------------------------- #

# -------------------------------------------------------------------------- #
#
# THIS COULD ALSO BE SHOWN ON A TIME SERIES PLOT
#
# ONLY SELECT THE ONES WITH THE GREATEST VALUES (SHAPE/COLOR A SPECIFIC WAY) 
# AND PLOT ON THE DAYS THAT THEY WERE PLAYED
#
# -------------------------------------------------------------------------- #

non_dt_complete_df = complete_df.copy()
# convert end_time to datetime object
complete_df['end_time'] = pd.to_datetime(complete_df['end_time'])

dt_complete_df = complete_df.set_index('end_time', inplace=False)

# -------------------------------------------------------------------------- #
#
# CREATE A SCATTERPLOT TO LINK WITH THE TIME SERIES (unable to complete)
#
# -------------------------------------------------------------------------- #
"""
valence_noise = px.scatter(
    complete_df.sort_values('end_time'),  # sort values to be plotted chronologically
    x='end_time',
    y='valence',
    color_discrete_sequence = px.colors.qualitative.Bold,
    title="Valence History (songs)",
    labels={"end_time": "Time", "valence": "Valence"},
    hover_name = 'track_name',
    hover_data = ['artist_name'] + measures
)
valence.update_layout(hovermode="x",
                      font_family = "Optima",
                      plot_bgcolor='#FFFFFF',
                      font_color = '#05204A',
                      height = 650,
                      title={
                            'y':0.95,
                            'x':0.5,
                            'font_size': 30,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      hoverlabel=dict(font_size=16,
                                      font_family="Optima"))




"""

time_series = go.Figure()
# scatter_series = go.Figure()
buttons = []
# buttons_scatter = []
for measure in measures:
    measure_col = dt_complete_df[[measure]].copy() # just the dates and the valence
    
    # plot the components of the time series data   
    measure_trend = (seasonal_decompose(measure_col, model="additive", 
                                               period = int(len(measure_col)/12))
                     .trend
                     .to_frame()
                     .reset_index())

    time_series.add_trace(go.Scatter(
                          x = measure_trend['end_time'],
                          y = measure_trend['trend'],
                          name = measure))
    
    # scatter_series.add_trace(go.Scatter(
        # x = measure_col.index,
        # y = measure_col[measure],
        # name = measure))
    
    # looks like you need to create a list of buttons. Each element is a dict
    button = dict(
        method = 'update',
        label = measure,
        args = [{'y': [measure_trend['trend']]}])
                 # 'x': [measure_trend['end_time']]}])
    buttons.append(button)
    
    # button_scatter = dict(
        # method = "update",
        # label = measure,
        # args = [{"y": [measure_col[measure]]}])
    # buttons_scatter.append(button_scatter)
    
    

time_series.update_xaxes(rangeslider_visible=True) 
# scatter_series.update_xaxes(rangeslider_visible = True)

time_series.update_layout(updatemenus=[dict(buttons=buttons, 
                                            direction='down', 
                                            x=0.1, 
                                            y=1.15)],
                          # hovermode="x",
                          font_family = "Optima",
                          plot_bgcolor='#FFFFFF',
                          font_color = '#05204A',
                          height = 650,
                          title={
                                'y':0.95,
                                'x':0.45,
                                'font_size': 30,
                                'xanchor': 'center',
                                'yanchor': 'top'})
                          #hoverlabel=dict(font_size=16,
                                      #font_family="Optima"))

# scatter_series.update_layout(updatemenus=[dict(buttons=buttons_scatter, 
                                            # direction='down', 
                                            # x=0.1, 
                                            # y=1.15)],
                             # hovermode="x",
                             # font_family = "Optima",
                             # plot_bgcolor='#FFFFFF',
                             # font_color = '#05204A',
                             # height = 650,
                             # title={
                                 # 'y':0.95,
                                 # 'x':0.45,
                                 # 'font_size': 30,
                                 # 'xanchor': 'center',
                                 # 'yanchor': 'top'},
                             # hoverlabel=dict(font_size=16,
                                      # font_family="Optima"))

# -------------------------------------------------------------------------- #
#
# CREATING THE MIN/MAX TREE MAP
#
# -------------------------------------------------------------------------- #

mins_maxes = pd.DataFrame()

for measure in measures:
    max_value = max(non_dt_complete_df[measure]) # find the max value
    row = non_dt_complete_df[non_dt_complete_df[measure] == max_value] # find rows that match the value
    row['min_max'] = ['max' for i in range(len(row))]
    row['selection'] = [measure for i in range(len(row))]
    
    mins_maxes= mins_maxes.append(row)
    
    
    max_songs = row['track_name'].unique()
    """
    for song in max_songs:
        print(song, 'was played on:')
        dates = row[row['track_name'] == song]['end_time']
        for date in dates:
            print(date)
    """
    summary_stats = ('The maximum value of ' + measure + ' is ' + str(round(max_value, 4))
                     + ' from the song(s):\n ' + '\n'.join(row['track_name'].unique()) + '\n\n\n'
                     + '\n'.join(row['track_name'].unique()) + ' was played at the following times:\n'
                     + '\n'.join(row['end_time']))
    
    min_value = min(non_dt_complete_df[measure]) # find the max value
    row = non_dt_complete_df[non_dt_complete_df[measure] == min_value] # find rows that match the value
    
    row['min_max'] = ['min' for i in range(len(row))]
    row['selection'] = [measure for i in range(len(row))]

    
    mins_maxes = mins_maxes.append(row)
    
    min_songs = row['track_name'].unique()
    """
    for song in min_songs:
        print(song, 'was played on:')
        dates = row[row['track_name'] == song]['end_time']
        for date in dates:
            print(date)
    """
    
    summary_stats += ('\n\nThe minimum value of ' + measure + ' is ' + str(round(min_value, 4))
                     + ' from the song(s):\n ' + '\n'.join(row['track_name'].unique()) + '\n\n\n'
                     + '\n'.join(row['track_name'].unique()) + ' was played at the following times:\n'
                     + '\n'.join(row['end_time']))
    
    
mins_maxes = mins_maxes.drop(['ms_played', 'acousticness', 'instrumentalness',
                          'liveness', 'loudness', 'speechiness',
                          'time_signature'], axis = 1)  

mins_maxes_melt = pd.melt(mins_maxes, 
                     id_vars = ['id', 'track_name', 'artist_name', 'end_time',
                                'min_max', 'selection'],
                     var_name = 'feature')

# drop the rows whose selection doesn't match feature
mins_maxes_melt_clean = mins_maxes_melt[mins_maxes_melt['selection'] == mins_maxes_melt['feature']]

# count how many times each song appears... merge based on the song_id
counts_listened = (complete_df.groupby(['id'])
                  .count()
                  .reset_index()
                  [['id','end_time']] # select two columns
                  .rename(columns = {'end_time': 'count'}) # rename to count
                  .sort_values('count', ascending = False) # rank from greatest to least
                  .reset_index(drop = True))

# df with counts now, merge based on song id
merge_min_max_count = mins_maxes_melt_clean.merge(counts_listened, how = 'left',
                                on = 'id')

merge_min_max_clean = merge_min_max_count[merge_min_max_count['feature'] != 'mode'] 
merge_min_max_clean = merge_min_max_clean[merge_min_max_count['feature'] != 'key'] 

# I think a treemap is the best way to show this information
min_max_tree = px.treemap(merge_min_max_clean,
                          title = "Maximum and Minimum Values for Select Features",
                          path=[px.Constant("Danceability, Energy, Tempo, and Valence"),
                                'feature',
                                'min_max',
                                'value',
                                'track_name', 
                                'end_time'],
                          values = 'value',
                          color = 'count',
                          hover_name = 'track_name',
                          hover_data = {
                              'min_max': True,
                              'value': True,
                              'count': True,
                              'end_time': True,
                              'artist_name': True},
                          color_continuous_scale = 'sunsetdark',
                          color_continuous_midpoint=np.average(merge_min_max_clean['count'],
                                                               weights=merge_min_max_clean['value']))
# may have to do something with clickmode/dragmode
min_max_tree.update_layout(uniformtext=dict(minsize=10, mode='hide'),
                           margin = dict(t=100, l=25, r=25, b=25),
                           font_family = 'Optima',
                           font_color = '#05204A',
                           title = {'y':0.95,
                                'x':0.45,
                                'font_size': 30,
                                'xanchor': 'center',
                                'yanchor': 'top'},
                           hoverlabel=dict(
                               font_size=16,
                               font_family = "Optima",
                               font_color = '#FFFFFF')
                           )
min_max_tree.data[0]['textfont']['color'] = "#FFFFFF"
min_max_tree.data[0]['hovertemplate'] = '<b>%{hovertext} by %{customdata[3]}</b><br><br>Value (%{customdata[0]}): %{customdata[1]}<br># Times Played: %{color}<extra></extra>'
min_max_tree.data[0]['hovertext'] = np.array(['Raincoat', 'Raincoat', 'Raincoat', 'Raincoat', 'Raincoat', 'Raincoat',
                         'LIGHT RAIN', 'LIGHT RAIN', 'Raincoat', 'Raincoat', 'Raincoat',
                         'Raincoat', 'Raincoat', 'Raincoat', 'What a Fool Believes',
                         'What a Fool Believes', 'Raincoat', 'Raincoat', 'Anybody But You',
                         '1 4 Me', 'Abracadabra', 'What a Fool Believes', 'What a Fool Believes',
                         'Raincoat', 'Raincoat', 'Summer Rain', 'What a Fool Believes',
                         'Raincoat', 'Raincoat', 'What a Fool Believes',
                         'Concerto in G major for Recorder, strings and continuo: 2. Adagio',
                         'Raincoat', 'Raincoat', 'Raincoat', 'Raincoat', '1 4 Me', 'Abracadabra',
                         'Anybody But You',
                         'Concerto in G major for Recorder, strings and continuo: 2. Adagio',
                         'LIGHT RAIN', 'LIGHT RAIN', 'Raincoat', 'Raincoat', 'Summer Rain',
                         'What a Fool Believes', '–', '1 4 Me', 'LIGHT RAIN', 'Abracadabra',
                         'What a Fool Believes', 'Raincoat',
                         'Concerto in G major for Recorder, strings and continuo: 2. Adagio',
                         'Anybody But You', 'Abracadabra', 'Raincoat', 'Anybody But You',
                         'What a Fool Believes', 'LIGHT RAIN', '1 4 Me',
                         'Concerto in G major for Recorder, strings and continuo: 2. Adagio',
                         '–', '–', '–', '–', '–', '–'])

min_max_tree.data[0]['customdata'] = np.array([['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 1.0, "WOOF! IT'S BILLY"],
       ['min', 0.0547, 1.0, "WOOF! IT'S BILLY"],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['max', 0.985, 6.0, 'The Doobie Brothers'],
       ['max', 0.985, 6.0, 'The Doobie Brothers'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['max', 211.489, 1.0, 'Malia Civetz'],
       ['min', 0.00102, 1.0, 'Electric Guest'],
       ['max', 0.983, 1.0, 'High Low Row'],
       ['max', 0.985, 6.0, 'The Doobie Brothers'],
       ['max', 0.985, 6.0, 'The Doobie Brothers'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 1.0, 'Good Boy Daisy'],
       ['max', 0.985, 6.0, 'The Doobie Brothers'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['max', 0.985, 6.0, 'The Doobie Brothers'],
       ['min', 41.698, 1.0, 'Tomaso Albinoni'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 11.0, 'Timeflies'],
       ['max', 1.0, 11.0, 'Timeflies'],
       ['min', 0.00102, 1.0, 'Electric Guest'],
       ['max', 0.983, 1.0, 'High Low Row'],
       ['max', 211.489, 1.0, 'Malia Civetz'],
       ['min', 41.698, 1.0, 'Tomaso Albinoni'],
       ['min', 1e-05, 1.0, "WOOF! IT'S BILLY"],
       ['min', 0.0547, 1.0, "WOOF! IT'S BILLY"],
       ['min', 0.00011, 11.0, 'Timeflies'],
       ['max', 11.0, 11.0, 'Timeflies'],
       ['min', 1e-05, 1.0, 'Good Boy Daisy'],
       ['max', 5.91, 6.0, 'The Doobie Brothers'],
       ['min', 0.00013000000000000002, 9.461538461538462, '–'],
       ['min', 0.00102, 1.0, 'Electric Guest'],
       ['min', 0.0547, 1.0, "WOOF! IT'S BILLY"],
       ['max', 0.983, 1.0, 'High Low Row'],
       ['max', 5.91, 6.0, 'The Doobie Brothers'],
       ['max', 11.0, 11.0, 'Timeflies'],
       ['min', 41.698, 1.0, 'Tomaso Albinoni'],
       ['max', 211.489, 1.0, 'Malia Civetz'],
       ['max', 0.983, 1.0, 'High Low Row'],
       ['max', 11.0, 11.0, 'Timeflies'],
       ['max', 211.489, 1.0, 'Malia Civetz'],
       ['max', 5.91, 6.0, 'The Doobie Brothers'],
       ['min', 0.0547, 1.0, "WOOF! IT'S BILLY"],
       ['min', 0.00102, 1.0, 'Electric Guest'],
       ['min', 41.698, 1.0, 'Tomaso Albinoni'],
       ['min', 0.00013000000000000002, 9.461538461538462, '–'],
       ['(?)', 1.0377, 1.0, '–'],
       ['(?)', 11.00102, 10.999072813248226, '–'],
       ['(?)', 253.187, 1.0, '–'],
       ['(?)', 5.91013, 6.000076140457149, '–'],
       ['(?)', 271.13585, 1.5146906984081965, '–']])

with open('fig-5c.html', 'w') as f:
    f.write(time_series.to_html(full_html=False, include_plotlyjs='cdn')) # perhaps the key to getting the arrays 
    # f.write(scatter_series.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(min_max_tree.to_html(full_html=False, include_plotlyjs='cdn'))
    
"""
pio.write_html(min_max_tree, 
               #full_html = False, 
               file="min_max_scatter.html", 
               div_id = 'treeplot',
               auto_open=True)
"""

# NOW find a way to connect the two graphs (min/max to the points on the time series)

    

    