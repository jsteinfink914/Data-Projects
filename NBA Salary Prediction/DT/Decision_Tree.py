# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:20:48 2021

@author: jstei
"""

# =============================================================================
# This code clusters text data from NBA twitter searches. The already cleaned
# outputs of the searches are stored as text files. The code uses
# CountVectorizer to create the data, kmeans clustering to cluster, as well as
# the elbow, silhouette, and gap statistic method to find optimal clusters.
# The code also creates a dendrogram, 3D images of the text clusters, and 
# wordclouds to help visualize.
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


##Creating document term matrix for each search
LargeDTM=pd.DataFrame()
for i in range(len(Files)):
    Content=[]
    DF=pd.read_csv(Files[i],header=None,index_col=False,encoding='cp1252')
    with open(Files[i],'r',encoding='cp1252') as file:
        next(file)
        for row in file:
            Content.append(row)
    CV=CountVectorizer(input='content',stop_words='english')
    DTM=CV.fit_transform(Content)
    ##Collecting vocab
    ColNames=CV.get_feature_names()
    ##Converting to pandas data frame
    DF1=pd.DataFrame(DTM.toarray(),columns=ColNames)
    ##Renaming data frame with labels
    Label=[list(topics.values())[i]]*len(DF)
    DF1=DF1.drop(Label[0],axis=1)
    Label=Label[:-1]
    DF1.insert(loc=0,column='LABEL',value=Label)
    Columns_new_data=list(DF1.columns)
    Columns_master_data=list(LargeDTM.columns)
    shared_columns=list(set(Columns_new_data).intersection(Columns_master_data))
    LargeDTM=pd.concat([LargeDTM,DF1])
        
    
##Want to drop some irrelevant numbers and words
ColNames=DF.columns
##Normalize the data
LargeDTM=LargeDTM.fillna(0)
#sum(LargeDTM.isna())
labels=LargeDTM.LABEL
LargeDTM=LargeDTM.drop('LABEL',axis=1)
LargeDTM=(LargeDTM-LargeDTM.min())/(LargeDTM.max()-LargeDTM.min())
LargeDTM.insert(loc=0,column='LABEL',value=labels)
LargeDTM.to_csv('Normalized_Labeled_Text_DT.csv',index=None)
