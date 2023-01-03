# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:17:50 2021

@author: jstei
"""
# =============================================================================
# This code takes the clean and normalized text data from the 
# previously performed twitter searches regarding NBA stardom 
# grouped into the three clusters(but the code can work for any
# labeled DF). The code creates wordclouds for each of the labels,
# creates and checks a train and test split using a 80/20 split,
# and then performs NB analysis making a confusion matrix and
# collecting feature log probabilities
# =============================================================================
##Importing necessary libraries
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
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import graphviz
from sklearn.decomposition import LatentDirichletAllocation 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics import classification_report
##Reading in the data and isolating labels
DF=pd.read_csv('Normalized_Labeled_Text_DT.csv')
Labels=DF.LABEL

##Creating wordckouds for each label
List_of_WC=[]

for topic in set(Labels):
    tempdf = DF[DF['LABEL'] == topic]
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    #print(tempdf)
    
    #Make var name
    wc = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4, #mask=next_image,
                   max_words=200).generate_from_frequencies(tempdf)
    
    ## Here, this list holds all three wordclouds I am building
    List_of_WC.append(wc)
    

##Plotting the wordclouds
NumTopics=len(set(Labels))
topics=list(set(Labels))
for i in range(NumTopics):
    fig=plt.figure(figsize=(25, 25))
    #ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.title(str.upper(topics[i]),fontsize=50)

###Naive Bayes

##Train test split of 80/20
TrainDF,TestDF=train_test_split(DF, test_size=.2)

##Checking if train and test sets are balanced
Trainvalues=[]
Testvalues=[]
for i in topics:
    Trainvalues.append(TrainDF[TrainDF.LABEL==i].shape[0])
    Testvalues.append(TestDF[TestDF.LABEL==i].shape[0])

##Plotting train and test set label counts in one plot
plt.subplot(1,2,1)
plt.bar(topics, Trainvalues, color ='maroon',width=.6)
plt.xlabel("Labels",fontsize=22,labelpad=15)
plt.xticks(size=20)
plt.ylabel("Count",fontsize=22)
plt.yticks(size=20)
plt.title("Label count in train set",fontsize=30)

plt.subplot(1,2,2)
plt.bar(topics, Testvalues, color ='maroon',width=.6)
plt.xlabel("Labels",fontsize=22,labelpad=15)
plt.xticks(size=20)
plt.ylabel("Count",fontsize=22)
plt.yticks(size=20)
plt.title("Label count in test set",fontsize=30)
plt.show()

##Dropping labels from train and test sets
Trainlabel=TrainDF.LABEL
TrainDF=TrainDF.drop('LABEL',axis=1)
Testlabel=TestDF.LABEL
TestDF=TestDF.drop('LABEL',axis=1)

##Initiating Naive Bayes
NB=MultinomialNB()
##Creating the NB model
model=NB.fit(TrainDF,Trainlabel)
##Predicting test values
predict=NB.predict(TestDF)

##Plotting confusion matrix using seaborn heatmap
Acc=confusion_matrix(Testlabel,predict)
fig=plt.figure()
sns.heatmap(Acc,annot=True,fmt='g',annot_kws={'size':20})
plt.xlabel('Predicted labels',fontsize=20)
plt.ylabel('True labels',fontsize=20)
plt.title('Confusion Matrix - Naive Bayes - Text',fontsize=30)
plt.xticks(ticks=[0.5,1.5,2.5],labels=topics,fontsize=15)
plt.yticks(ticks=[0.5,1.5,2.5],labels=topics,fontsize=15)

##Looking at classification report
Acc1=classification_report(Testlabel,predict)
print(Acc1)

##Isolating and plotting feature log probabilities
feature_prob=NB.feature_log_prob_
feature_names=TrainDF.columns
##Looping through each label and sorting log probabilities
for i in range(len(topics)):
    feature_probs=feature_prob[i]
    indices=np.argsort(feature_probs)[::-1]
    ##Plotting subplots for each label
    plt.subplot(1,3,i+1)
    plt.barh(feature_names[indices[0:20]], feature_probs[indices[0:20]],
             color ='blue')
    if i==0:
        plt.ylabel("Feature",fontsize=15)
    plt.xlabel("Feauture Probabilities",fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    title='Feature Probabilities: '+str(topics[i])
    plt.title(title,fontsize=20)
    plt.subplots_adjust(wspace=1)
    plt.show()
    
    
    
    
