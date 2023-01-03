import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas

# =============================================================================
# Converting Country from 2 letter abreviation to 3
# =============================================================================

##Dataframe with 3 letter country code
codes_df = pd.read_csv("countries_codes_and_coordinates.csv")

##Data with 2 letter country code and accessibility data
DF_counts=pd.read_csv('Map.csv')


# country, count, geometry
# choropleth
# https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

##Making a dictionary with 2 letter country code as key and 3 letter as value
code_dict = {}
for index, row in codes_df.iterrows():
    code_dict[row["Alpha-2 code"][2:4]] = row["Alpha-3 code"][2:5]
    

##Looping through data and replacing 2 letter country code with 3
for i in range(len(DF_counts)):
    code = DF_counts.loc[i,'Country']
    if code not in code_dict.keys():
        continue
    else:
        DF_counts.loc[i,'Country'] = code_dict[code]
# country, count, geometry
# choropleth
# https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html


# =============================================================================
# Making map
# =============================================================================

##Reading in default world map
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world = world[['iso_a3','geometry']]

##Merging location data with accessibility data
world1 = world.merge(DF_counts,left_on='iso_a3',right_on='Country',how='outer')



##Plotting the map
fig,ax = plt.subplots(1, 1)

world1.plot(column='Percentage', ax=ax, legend=True,cmap='OrRd', missing_kwds={
        "color": "lightgrey",
        "label": "Missing values"},
        legend_kwds={'label': "Proportion of Top Songs Available in Each Country",
                        'orientation': "horizontal"})
plt.title('Spotify Accessibility Around the World')
plt.xticks(color='w')
plt.yticks(color='w')
plt.savefig('fig-1_WorldMap.png')
        


# https://geopandas.org/en/stable/gallery/create_geopandas_from_pandas.html
# https://python-graph-gallery.com/choropleth-map-geopandas-python
