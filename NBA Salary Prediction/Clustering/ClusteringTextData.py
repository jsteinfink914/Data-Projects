# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:24:31 2021

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
path="C:/Users/jstei/Desktop/ANLY_501/Twitter"
##Collecting full file names
Files= [path + "/" + file for file in os.listdir(path)]
##Collecting topic names
Topics = [file.split('.')[0] for file in os.listdir(path)]
##Collecting labels and editing out the unnecessary'nba_'
topics={}
for i in range(len(Topics)):
    name=Topics[i].split('_')[1]
    topics[i]=name

##Initializing count vectorizer
CV=CountVectorizer(input='filename',stop_words='english',encoding='cp1252')
##Creating document term matrix
DTM=CV.fit_transform(Files)
##Collecting vocab
ColNames=CV.get_feature_names()

##Converting to pandas data frame
DF=pd.DataFrame(DTM.toarray(),columns=ColNames)

##Renaming data frame with labels
DF=DF.rename(topics,axis='index')
##Want to drop some irrelevant numbers and words
DF=DF.drop(ColNames[:47],axis=1)
ColNames=DF.columns
##Normalize the data
DF=(DF-DF.min())/(DF.max()-DF.min())
DF.to_csv('NormalizedText.csv',index=None)


# =============================================================================
# CLUSTERING
# First, will look at the elbow, silhouette, and gap statistic methods
# to determine optimal clusters or optimal cluster range
# =============================================================================

##ELBOW METHOD
k={}
##Checking up to 7 clusters and collecting sum of squares
for i in range(1,6):
    kmeans=KMeans(n_clusters=i).fit(DF)
    labels=kmeans.labels_
    k[i]=kmeans.inertia_
##Plotting number of clusters against sum of squares
plt.figure()
plt.plot(k.keys(),k.values())
plt.title('Elbow Method for Text')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()
##Plot shows that there is an optimal number of clusters is 2

##SILHOUETTE METHOD
for i in ['euclidean','manhattan','cosine']:
    Sil={}
    for k in range(2,5):
        kmeans=KMeans(n_clusters=k)
        model=kmeans.fit(DF)
        prediction=kmeans.predict(DF)
        labels_n=kmeans.labels_
        score=metrics.silhouette_score(DF,labels_n,metric=i)
        Sil[k]=score
    
    plt.figure()
    plt.plot(Sil.keys(),Sil.values())
    plt.title('Silhouette: '+i)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Coefficient')
    plt.show()
##Maximum is at 2 clusters as well for euclidean and manhattan distance
##Cosine similarity yielded negative silhouette coefficients and thus 
##is likely a bad distance measure to use for this data

##GAP STATISTIC
##Creating array of 0's to store gaps
gaps = np.zeros(len(range(1,6)),)
##Empty dataframe to store results
resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
for gap_index, k in enumerate(range(1,6)):
    # Holder for reference dispersion results
    referenceDisps = np.zeros(20)
    # For n references, generate random sample and perform kmeans 
    # getting resulting dispersion of each loop
    # This for loop is creating the 'null reference distribution' that can
    # be compared to the actual distribution
    for i in range(20):
        # Create new random reference set
        randomReference = np.random.random_sample(size=DF.shape)
        # Fit to it
        km = KMeans(k)
        km.fit(randomReference)
        refDisp = km.inertia_
        referenceDisps[i] = refDisp
    # Fit cluster to original data and create dispersion
    km = KMeans(k)
    km.fit(DF)
    originalDisp = km.inertia_
    # Calculate gap statistic
    gap = np.log(np.mean(referenceDisps)) - np.log(originalDisp)
    # Assign this loop's gap statistic to gaps
    gaps[gap_index] = gap
    resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
##Plotting results
plt.plot(resultsdf['clusterCount'], resultsdf['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic vs. K')
plt.show()
##Gap statistic shows that 4 clusters is optimal




##Initialize kmeans with differing number of clusters (2-4)
for i in range(2,5):
    kmeans=KMeans(n_clusters=i)
    ##Fit to DF
    kmeans.fit(DF)
    ##Collecting clusters
    labels=kmeans.labels_
    #print(labels)
    ##Printing centroids
    centroids=kmeans.cluster_centers_
    ##predicting clusters to validate
    prediction_kmeans=kmeans.predict(DF)
    #print(prediction_kmeans)
    Results=pd.DataFrame([DF.index,labels]).T
    print('Results for '+str(i)+' clusters')
    print(Results)
    print('\n')
    ##Building wordcloud for 2 clusters
    if i==3:
        for lab in set(labels):
            # list of words in cluster
            words = []
            #indexes of df in cluster
            indexes = np.array(Results[Results[1] == lab].index)
            #df only containing one cluseter
            temp_df = DF.loc[DF.index.values[indexes],]
            for b in range(len(ColNames)):
                #number of one word in single column
                words_count = int(sum(temp_df[ColNames[b]]))
                #list of same word
                lst = [ColNames[b] for j in range(words_count)]
                #add list to word list
                words += lst
            #join words together into single string
            words = " ".join(words)
            #plot wordcloud
            wordcloud = WordCloud(collocations=False,max_words=1000000,stopwords='english').generate(words)
            plt.figure(figsize = (10,15))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title("Cluster" + str(lab+1))
            plt.show()
    ##Making 3D Viz for the cluster centers and the chosen keywords for 2 clusters
        for i in ['allstar','mvp','star']:
            indexes=DF.columns.get_loc(i)
            #print(indexes)
        x=DF["allstar"]   #column 203
        y=DF["mvp"] #column 5184
        z=DF["star"]  #column 7532
        fig1=plt.figure()
        ax1=Axes3D(fig1,rect=[0,0,.9,1],elev=48,azim=134)
        ##Plotting the keywords
        ax1.scatter(x,y,z,cmap='RdYlGn',edgecolor='k', s=200,c=prediction_kmeans)
        ax1.w_xaxis.set_ticklabels([])
        ax1.w_yaxis.set_ticklabels([])
        ax1.w_zaxis.set_ticklabels([])
        ax1.set_xlabel('allstar', fontsize=25)
        ax1.set_ylabel('mvp', fontsize=25)
        ax1.set_zlabel('star', fontsize=25)
        ##Collecting centroids of the key words and plotting those 
        C1=centroids[0,(203,5184,7532)]
        #print(C1)
        C2=centroids[1,(203,5184,7532)]
        #print(C2)
        C3=centroids[2,(203,5184,7532)]
        xs=C1[0],C2[0],C3[0]
        #print(xs)
        ys=C1[1],C2[1],C3[1]
        zs=C1[2],C2[2],C3[2]
        ##Plotting the cluster centroids
        ax1.scatter(xs,ys,zs, c='black', s=2000, alpha=0.2)
        plt.show()
##The star category seems to be the most unique while allstar and superstar are
##linked together, along with legend

##Using DBSCAN
MyDBSCAN = DBSCAN(eps=50, min_samples=2)
## eps is the maximum distance between two samples for 
## one to be considered as in the neighborhood of the other.
## This data has so few points that epsilon had to be massive
##In this case DBSCAN was not a helpful method
MyDBSCAN.fit_predict(DF)
FitDBSCAN=DBSCAN(eps=50,min_samples=2).fit(DF)
labels=MyDBSCAN.labels_

print(MyDBSCAN.labels_)
DBResults=pd.DataFrame([DF.index,labels]).T

##Similar results as kmeans with star on its own but most closely associated
##with mvp while allstar legend and superstar fall into the same group

##Hierarchical clustering
for i in ['euclidean','manhattan','cosine']:
    if i=='euclidean':
        HC=AgglomerativeClustering(n_clusters=2,affinity=i,linkage='ward')
    else:
        HC=AgglomerativeClustering(n_clusters=2,affinity=i,linkage='average')
    Fit=HC.fit(DF)
    HC_labels=HC.labels_
    print(HC_labels)
    plt.figure(figsize=(12,12))
    plt.title('Hierarchical Clustering')
    dendrogram=hc.dendrogram((hc.linkage(DF,method='ward')),
                             labels=[i for i in topics.values()])



    

    
