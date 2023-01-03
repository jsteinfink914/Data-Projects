# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:51:55 2022

@author: jstei
"""


import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from itertools import product
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


# =============================================================================
# HeatMap
# =============================================================================

data = pd.read_csv("TopSongs_clean.csv")
data2 = data.drop(["title", "top genre", "year", "artist"], axis=1)

##Getting correlations and making a numpy array
corr = data2.corr()
np_corr = corr.to_numpy().round(3)

##Plotting the heatmap
fig5 = ff.create_annotated_heatmap(
    z=np_corr,
    x=data2.columns.tolist(),
    y=data2.columns.tolist(),
    colorscale=px.colors.diverging.RdYlBu,
    # hoverinfo="none", #Shows hoverinfo for null values
    showscale=True,
    ygap=1,
    xgap=1,
)
fig5.update_xaxes(side="bottom")

fig5.update_layout(
    title_text="Correlations Between Song Characteristics",
    title_x=0.5,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    yaxis_autorange="reversed",
    template="plotly_white",
    width=800,
    height=600,
    margin=dict(t=100)
)
fig5.write_html("fig-2_Heatmap.html")


# =============================================================================
# Trendlines of Attributed With Dropdown
# =============================================================================

##Grouping by genre and year and taking the mean of the attributes
grouped_data = data.groupby(by=["top genre", "year"]).mean().reset_index()

##Collecting the years
date = grouped_data["year"].unique()
date.sort()

##Attributes of interest and can be plotted on the same axis
attributes = [
    "energy",
    "danceability",
    "liveliness",
    "valence",
    "acousticness",
    "speechiness",
    "popularity",
]

##Top genres to be visualized
genre_list = ["boy band", "edm", "hip hop", "house", "pop", "rap"]


##Instantiate the figure
fig = go.Figure()

##For each genre
for genre in genre_list:
    ##And each attribute
    for attribute in attributes:
        ##Add a trace of the mean of the attribute and genre over time
        fig.add_trace(
            go.Scatter(
                x=grouped_data["year"][grouped_data["top genre"] == genre],
                y=grouped_data[attribute][grouped_data["top genre"] == genre],
                name=attribute,
                visible=True,
            )
        )

##Empty list to store the buttons
buttons = []

##Creating list of all combinations of genres and attributes
all_combinations = product(genre_list, attributes)
all_combinations = [i for i in all_combinations]


for i, genre in enumerate(genre_list):
    ##Establish visibility as false for all traces
    args = [False] * len(all_combinations)
    ##Assign True for visibility of traces for each genre
    args[i * len(attributes) : len(attributes) * (i + 1)] = [True] * len(attributes)
    ##Make sure the list does not change lengths--not sure why this happens
    while len(args) < len(all_combinations):
        args.append(False)

    ##Creating the button with the genre as the name
    button = dict(label=genre, method="update", args=[{"visible": args}])

    buttons.append(button)

##Assigning buttons to the layout
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            type="dropdown",
            buttons=buttons,
            x=0,
            y=1.1,
            xanchor="left",
            yanchor="bottom",
        )
    ],
    autosize=True,
    title=dict(text="Attribute Trends by Genre", x=0.5),
    xaxis=dict(title="Year"),
    yaxis=dict(title="Attribute Mean"),
)


fig.write_html("fig-3_TimeSeries.html")

# =============================================================================
# Working Bar Chart with dropdown
# =============================================================================


##Second View
fig2 = go.Figure()

colors=['blue','black','red','green','pink','orange']
##For each genre add the bar chart of the top artists
for i in range(len(genre_list)):
    fig2.add_trace(
        go.Bar(
            x=data[data["top genre"] == genre_list[i]]["artist"].value_counts().index,
            y=list(data[data["top genre"] == genre_list[i]]["artist"].value_counts()),
            name=genre,
            visible=True,
            marker_color=colors[i]
        )
    )

buttons = []

##Create buttons such that each view gets shown only when the genre is selected
for i, genre in enumerate(genre_list):
    args = [False] * len(genre_list)
    args[i] = True

    button = dict(label=genre, method="update", args=[{"visible": args}])

    buttons.append(button)

fig2.update_layout(
    updatemenus=[
        dict(
            active=0,
            type="dropdown",
            buttons=buttons,
            x=0,
            y=1.1,
            xanchor="left",
            yanchor="bottom",
        )
    ],
    autosize=True,
    title=dict(text="Artists with the most Top Songs (2010-2019)", x=0.5),
    xaxis=dict(title="Artist"),
    yaxis=dict(title="Number of Songs"),
)

# Part 1 of linked view, can uncomment to generate plot
#fig2.write_html("TopArtists.html")



# =============================================================================
# Looking at distributions of duration, loudness, and bpm
# =============================================================================
fig4 = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Beats Per Minute", "Duration", "Decibels (Loudness)"))

for genre in genre_list:
    fig4.add_trace(
        go.Histogram(
            x=data[data["top genre"] == genre]["bpm"],
            name='bpm',
            visible=True)
        ,row=1,col=1)
    if genre=='rap':
        fig4.add_trace(
            go.Histogram(
                x=data[data["top genre"] == genre]["duration"],
                name='duration',
                visible=True,
                nbinsx=3)
            ,row=1,col=2)
    else:
        fig4.add_trace(
            go.Histogram(
                x=data[data["top genre"] == genre]["duration"],
                name='duration',
                visible=True)
            ,row=1,col=2)
    fig4.add_trace(
        go.Histogram(
            x=data[data["top genre"] == genre]["Loudness"],
            name='pop',
            visible=True)
        ,row=1,col=3)
    
buttons=[]

##Create buttons such that each view gets shown only when the genre is selected
for i, genre in enumerate(genre_list):
    args = [False] * len(genre_list)*3
    args[i*3],args[(i*3+1)],args[(i*3+2)] = True,True,True

    button = dict(label=genre, method="update", args=[{"visible": args}])

    buttons.append(button)

fig4.update_layout(
    updatemenus=[
        dict(
            active=0,
            type="dropdown",
            buttons=buttons,
            x=0,
            y=1.1,
            xanchor="left",
            yanchor="bottom",
        )
    ],
    autosize=True,
    title=dict(text="Distributions of Bpm,duration, and dB by Genre", x=0.5),
    yaxis=dict(title="Frequency"),
)

# Part 2 of linked view, can uncomment to generate plot
#fig4.write_html("Distributions.html")




# =============================================================================
#  Linked View Attempt - Linking Distributions chart with Top Artists
# =============================================================================

fig5 = make_subplots(rows=2, cols=3,specs=[[{}, {}, {}],[{'colspan': 3}, None, None]],
                      subplot_titles=("Beats Per Minute", "Duration", "Decibels (Loudness)","Top Artists"))


##For each genre add the bar chart of the top artists
for i in range(len(genre_list)):
    fig5.add_trace(
        go.Bar(
            x=data[data["top genre"] == genre_list[i]]["artist"].value_counts().index,
            y=list(data[data["top genre"] == genre_list[i]]["artist"].value_counts()),
            name=genre_list[i],
            visible=True,
            marker_color=colors[i]),row=2,col=1)
    fig5.add_trace(
        go.Histogram(
            x=data[data["top genre"] == genre_list[i]]["bpm"],
            name='bpm',
            visible=True)
        ,row=1,col=1)
    if genre=='rap':
        fig5.add_trace(
            go.Histogram(
                x=data[data["top genre"] == genre_list[i]]["duration"],
                name='duration',
                visible=True,
                nbinsx=3)
            ,row=1,col=2)
    else:
        fig5.add_trace(
            go.Histogram(
                x=data[data["top genre"] == genre_list[i]]["duration"],
                name='duration',
                visible=True)
            ,row=1,col=2)
    fig5.add_trace(
        go.Histogram(
            x=data[data["top genre"] == genre_list[i]]["Loudness"],
            name='decibels',
            visible=True)
        ,row=1,col=3)

buttons = []

##Create buttons such that each view gets shown only when the genre is selected
for i, genre in enumerate(genre_list):
    args = [False] * len(genre_list)*4
    args[i*4],args[(i*4+1)],args[(i*4+2)],args[(i*4+3)] = True,True,True,True

    button = dict(label=genre, method="update", args=[{"visible": args}])

    buttons.append(button)

fig5.update_layout(
    updatemenus=[
        dict(
            active=0,
            type="dropdown",
            buttons=buttons,
            x=0,
            y=1.1,
            xanchor="left",
            yanchor="bottom",
        )
    ],
    autosize=True,
    title=dict(text="Song Characteristics and Top Artists by Genre", x=0.5)
)


fig5.write_html("fig-4_LinkedView.html")



