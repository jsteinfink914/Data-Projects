# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:20:48 2021

@author: jstei
"""

# =============================================================================
# This code prepares text data from NBA twitter searches for decision 
# tree analysis. The already cleaned outputs of the searches are stored as 
# text files. The code uses CountVectorizer to create the Document Term Matrix,
# for each search and merges them to make one master DTM to use for DT. 
# It normalizes the DTM and writes it to a csv file.
# =============================================================================
##Importing Necessary Libraries
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
import re
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

##Establishing path to text files
path="C:/Users/jstei/Desktop/ANLY_501/Twitter_csv_files/Clean_searches"
##Collecting full file names
Files= [path + "/" + file for file in os.listdir(path)]
##Collecting topic names
Topics = [file.split('.')[0] for file in os.listdir(path)]
##Collecting labels and editing out the unnecessary'nba_'
topics={}
for i in range(len(Topics)):
    name=Topics[i].split('_')[1]
    topics[i]=name


  ##List to store the tweet contents
Content=[]
LabelList=[]
for i in range(len(Files)):
    #opening file and skipping first row
    with open(Files[i],'r',encoding='cp1252') as file:
        next(file)
        x=0
        ##Appending 1000 tweets to Content list
        for row in file:
            Content.append(row)
            x+=1
            if x>=1000:
               break
    ##Creating label list to assign to dataframe later
    Label=[list(topics.values())[i]]*1000
    LabelList.append(Label)
LabelList=[i for sublist in LabelList for i in sublist]
##Creating DTM for each search where each tweet is a row
CV=CountVectorizer(input='content',stop_words='english')
DTM=CV.fit_transform(Content)
##Collecting vocab
ColNames=CV.get_feature_names()
##Converting to pandas data frame
LargeDTM=pd.DataFrame(DTM.toarray(),columns=ColNames)
##Drop search terms from DTM
search_terms=list(set(LabelList))
search_terms.append('allstar')
search_terms.append('superstar')
LargeDTM=LargeDTM.drop(search_terms,axis=1)
##Dropping one value from label list to account for difference
##between Label list length and DTM length
##inserting Label
LargeDTM.insert(loc=0,column='LABEL',value=LabelList)


        
    
ColNames=LargeDTM.columns
##subsititue missing 'NAN' values with 0
#sum(LargeDTM.isna())
#Dropping labels to normalize data
labels=LargeDTM.LABEL
LargeDTM=LargeDTM.drop('LABEL',axis=1)
##Mix-max normalization
LargeDTM=(LargeDTM-LargeDTM.min())/(LargeDTM.max()-LargeDTM.min())
##Adding back Label column
LargeDTM.insert(loc=0,column='LABEL',value=labels)
##Writing to csv
LargeDTM.to_csv('Normalized_Labeled_Text_DT.csv',index=None)
