#!/usr/bin/env python
# coding: utf-8

# # Capstone Project - Hypothetical Neighborhood Survery for B2B 
# ## Brief Description: This project seeks to identify places where a proposed IT managed service provider (The Company) wants to offer b2b IT systems, networks, and data management solutions to small to medium sized businesses.  

# In[1]:


#libraries installed
get_ipython().system('pip install geopy')
get_ipython().system('pip install pandas ')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install requests')
get_ipython().system('pip install kmeans')
get_ipython().system('pip install folium')
get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install soupsieve')
get_ipython().system('pip install lxml')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install json_flatten')
print ("libraries installed")


# In[2]:


# import all necessary libraries
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

# !conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

from bs4 import BeautifulSoup #scrape web page
import requests # library to handle requests
#from flatten_json import flatten # flatten nested json
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt 

# import k-means from clustering stage
from sklearn.cluster import KMeans

# !conda install -c conda-forge folium=0.5.0 --yes 
import folium # map rendering library
from folium.plugins import MarkerCluster


print('Imports complete.')


# In[3]:


# Using bs4 with CSS selectors to find and scape data from table on web

url1 = "https://en.wikipedia.org/wiki/List_of_highest-income_ZIP_Code_Tabulation_Areas_in_the_United_States"
soup = BeautifulSoup(requests.get(url1).content, "html.parser")

tbl = soup.select("h2:has(#ZCTAs_ranked_by_per_capita_income) + table")
df_zip_income = pd.read_html(str(tbl))[0]
print("df_zip_income created")


# In[4]:


#organize column headers
df_zip_income.columns = ['rank', 'place', 'zip', 'population', 'per_cap_inc']
df_zip_income.head()


# In[5]:


#load zipcode csv from github into dataframe
url2 = "https://raw.githubusercontent.com/adhorvitz/IBM_Coursera_Capstone/main/uszips_mod_zipgeoonly.csv"
df_zip_geo = pd.read_csv(url2, index_col=0)
df_zip_geo.head()


# In[6]:


#merge dataframes together
df_zip_good = pd.merge(df_zip_geo, df_zip_income, how = 'left', left_on = 'zip', right_on = 'zip')
print (df_zip_good.shape)


# In[7]:


#sort values
df_zip_good.sort_values(by=['rank'], ascending = False, inplace = False)
#strip values with "nan"
df_zip_top_income = df_zip_good[df_zip_good['rank'].notna()]
df_zip_top_income.head()


# In[9]:


#final df to be used that combines all necessary information (i.e. zip cross listed with lng, lat, and per captia)
df_income_zip_good = df_zip_top_income.filter(['zip', 'lat', 'lng', 'rank', 'place', 'population', 'per_cap_inc'], axis =1)
df_income_zip_good = df_income_zip_good.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
df_income_zip_good.columns = ['old_index','zip', 'lat', 'lng', 'rank','place', 'population', 'per_cap_inc']
df_income_zip_good.head()


# In[10]:


#check output
#print (df_income_zip_good.head(1000))
print (df_income_zip_good.dtypes)
print (df_income_zip_good.shape)


# In[11]:


#render data set visually into map to display locations
lati = 37.09024
longi =  -95.712891
map_usa = folium.Map(location=[lati, longi], zoom_start = 5)
for i, series in df_income_zip_good.iterrows():
    lat = series ['lat']
    lng = series ['lng']
    town = series ['place']

    folium.Marker (location=[lat,lng], popup = town, icon = folium.Icon(color='blue')).add_to(map_usa)
map_usa


# In[12]:


#create placeholder dataframe with HQ and plot onto Map
#geocoordinates for hq to compare with client requirements
datahq = [[41.1489, -73.9830, "Rockland HQ"]]
df_hq_loc = pd.DataFrame(datahq, columns = ['lat', 'lng', 'place'])
#latihq = 41.1489
#longihq = -73.9830
#map_hq = folium.map (location=[latihq,longihq], zoom_start = 5)
#folium.Marker (location=[latiq,longihq], popup = ("HQ"), icon = folium.Icon(color='yellow')).add_to(map_usa)
df_hq_loc


# In[13]:


map_hq = folium.Map(location=[lati, longi], zoom_start = 5)
for i, series in df_hq_loc.iterrows():
    lat = series ['lat']
    lng = series ['lng']
    town = series ['place']
    folium.Circle(location=[lat,lng], radius = 32186.9, color = 'orange').add_to(map_usa)
#map_usa
for i, series in df_hq_loc.iterrows():
    lat = series ['lat']
    lng = series ['lng']
    town = series ['place']
    folium.Circle(location=[lat,lng], radius =  72420.5, color = 'orange').add_to(map_usa)
map_usa


# In[14]:


#following locations meet criteria set forth in reserach problem / question
# pos_loc = possible locations
df_pos_loc = (df_income_zip_good.loc[[12,13,14]])
df_pos_loc.head()


# In[15]:


# reset index on df
df_pos_loc_good = df_pos_loc.reset_index (level=None, drop=False, col_level=0, col_fill ='')
df_pos_loc_good


# In[16]:


# modify marker lable for map display
df_pos_loc_marker = df_pos_loc_good.filter (['zip', 'rank', 'place', 'per_cap_inc'], axis = 1)
df_pos_loc_marker.head()


# In[17]:


#towns selected for final consideration as specified by research question mapped - locations marked in blue
lati = 40.8761622
longi = -74.240092
map_pos_loc = folium.Map(location=[lati, longi], zoom_start = 10)
for i, series in df_pos_loc_good.iterrows():
    lat = series ['lat']
    lng = series ['lng']
    town = series ['place']

    folium.Marker (location=[lat,lng], popup = (town), icon = folium.Icon(color='blue')).add_to(map_pos_loc)
    folium.Circle(location=[lat,lng], radius = 8046.72, color = 'orange').add_to(map_pos_loc)
map_pos_loc


# In[18]:


#load competitor data - probable competing businesses in selected locations
url_comp = "https://raw.githubusercontent.com/adhorvitz/IBM_Coursera_Capstone/main/competitors_new.csv"
df_competitors = pd.read_csv(url_comp)
df_competitors.head()


# In[19]:


#red for competitors - need something to loop and put in only venue name and category in popup
for lat, lng, cat, name in zip (df_competitors['lat'],df_competitors['lng'],df_competitors['categories'], df_competitors['name']):
    folium.CircleMarker(
        [lat, lng],
        radius=1,
        color='black',
        popup= (name, cat),
        fill = True,
        fill_color='black',
        fill_opacity=0.6
    ).add_to(map_pos_loc)
    #display map
map_pos_loc


# In[20]:


# @hidden_cell
# specifying credentials for Foursquare
CLIENT_ID = 'YBCSZCXFDWXUBOO33BVHOIVFNMKTNZD3LF3KBIHVTGU0BGIJ'
CLIENT_SECRET = 'GYEF12HHL5DCUC13WWY3JJSIHRMHE5IP1L45HIHQQH0GTQ4V'
VERSION = '20180604'


# In[21]:


def getNearbyVenues(names, latitudes, longitudes, radius=10000):
    
    venues_list=[]
    LIMIT = 500
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
       # print(type(results))
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])
    print(venues_list)
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
   
    return(nearby_venues)

df_venues = getNearbyVenues(names=df_pos_loc_good['place'],
                                   latitudes=df_pos_loc_good['lat'],
                                   longitudes=df_pos_loc_good['lng'])


# In[22]:


print (df_venues.shape)
df_venues.head()


# In[23]:


df_venues.groupby('Neighborhood').count()


# In[24]:


print ("there are {} unique categories.".format(len(df_venues['Venue Category'].unique())))


# In[25]:


#potential clients in orange
for lat, lng, name, cat in zip (df_venues['Venue Latitude'],df_venues['Venue Longitude'],df_venues['Venue'],df_venues['Venue Category']):
    folium.CircleMarker(
        [lat, lng],
        radius=1,
        color='red',
        popup= (name),
        fill = True,
        fill_color='red',
        fill_opacity=0.6
    ).add_to(map_pos_loc)
    #display map
map_pos_loc


# In[26]:


#one hot encoding - convert categorical into dummy / indicator variable
df_venues_onehot = pd.get_dummies(df_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
df_venues_onehot['Neighborhood'] = df_venues['Neighborhood']

#move neighborhood column to first column for ease of inspection
fixed_columns_venues = [df_venues_onehot.columns[-1]] + list(df_venues_onehot.columns[:-1])
df_venues_onehot = df_venues_onehot[fixed_columns_venues]

df_venues_onehot.head()


# In[27]:


df_venues_onehot.shape


# In[28]:


df_venues_grouped = df_venues_onehot.groupby('Neighborhood').mean().reset_index()
df_venues_grouped


# In[29]:


df_venues_grouped.shape


# In[30]:


num_top_venues = 5
for hood in df_venues_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = df_venues_grouped[df_venues_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[31]:


#reoganize the different categories per towns back into a pandas dataframe
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[32]:


#new dataframe to display top 10 venues for each location
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = df_venues_grouped['Neighborhood']

for ind in np.arange(df_venues_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(df_venues_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# Cluserting - k-means, unsupervised machine learning for (categorical) data

# In[33]:


# set number of clusters
kclusters = 3

df_venues_grouped_clustering = df_venues_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_venues_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[34]:


df_venues_grouped_clustering


# In[35]:


df_pos_loc_good


# In[36]:


df_pos_loc_good_stripped = df_pos_loc_good.drop (['index', 'old_index'], 1)
df_pos_loc_good_stripped


# In[37]:


df_pos_loc_good_reorg = df_pos_loc_good_stripped [['place', 'lat', 'lng', 'zip', 'population', 'per_cap_inc', 'rank']]


# In[38]:


df_pos_loc_good_reorg = df_pos_loc_good_reorg.rename(columns = {"place": "Neighborhood"})
df_pos_loc_good_reorg


# In[39]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
neighborhoods_venues_sorted


# In[40]:


df_venues_grouped_merged = df_pos_loc_good_reorg

# merge df_venues_grouped_grouped with df_pos_loc_good_data to add latitude/longitude for each neighborhood
df_venues_grouped_merged = df_venues_grouped_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

df_venues_grouped_merged.head() # check the last columns!


# In[41]:


# create map
map_clusters = folium.Map(location=[lat, lng], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lng, poi, cluster in zip(df_venues_grouped_merged['lat'], df_venues_grouped_merged['lng'], df_venues_grouped_merged['Neighborhood'], df_venues_grouped_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_pos_loc)
       
map_pos_loc

