# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:44:53 2022

@author: jstei
"""

import pandas as pd


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# =============================================================================
# Top Songs Dataset Reading and Cleaning.
# =============================================================================

data = pd.read_csv("TopSongs_raw.csv", encoding="ISO-8859-1")
data.shape
data.info()
data.head()
data.describe()
data.drop(["Unnamed: 0"], axis=1, inplace=True)

genres = {}
for i in data["top genre"].unique():
    genres[i] = data[data["top genre"] == i]["top genre"].count()
{k: v for k, v in sorted(genres.items(), key=lambda item: item[1])}

"""
Going to combine a bunch of subgenres.
There are many subtypes of:
   pop , hip hop, rap, edm, r&b, house

"""

for i in range(len(data)):
    if "pop" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "pop"
    elif "hip hop" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "hip hop"
    elif "rap" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "rap"
    elif "edm" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "edm"
    elif "r&b" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "r&b"
    elif "big room" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "house"
    elif "electro" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "edm"
    elif "complextro" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "edm"
    elif "tropical house" in data.loc[i, "top genre"]:
        data.loc[i, "top genre"] = "house"

data.rename(
    columns={
        "nrgy": "energy",
        "dnce": "danceability",
        "dB": "Loudness",
        "live": "liveliness",
        "val": "valence",
        "dur": "duration",
        "acous": "acousticness",
        "spch": "speechiness",
        "pop": "popularity",
    },
    inplace=True,
)

data.to_csv("TopSongs_clean.csv", index=False)
