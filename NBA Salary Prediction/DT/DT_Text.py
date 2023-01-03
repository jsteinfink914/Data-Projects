# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:45:20 2021

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


DF=pd.read_csv('Normalized_Labeled_Text_DT.csv')
Labels=DF.LABEL

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

###Decision Trees
TrainDF, TestDF=train_test_split(DF,test_size=.25,random_state=1)
Trainvalues=[]
Testvalues=[]
for i in topics:
    Trainvalues.append(TrainDF[TrainDF.LABEL==i].shape[0])
    Testvalues.append(TestDF[TestDF.LABEL==i].shape[0])

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

##Remove Labels from test data
TestLabels=TestDF['LABEL']
TestDF=TestDF.drop('LABEL',axis=1)
TrainLabels=TrainDF['LABEL']
TrainDF=TrainDF.drop('LABEL',axis=1)

##Decision Tree 1
DT1=DecisionTreeClassifier(criterion='gini',min_samples_split=10,
                           ccp_alpha=.003,splitter='best')
DT1.fit(TrainDF,TrainLabels)
feature_names=TrainDF.columns
Tree_Object=tree.export_graphviz(DT1,out_file=None,feature_names=feature_names,
                                 class_names=topics,filled=True,rounded=True,
                                 special_characters=True)
graph=graphviz.Source(Tree_Object)
graph.format='png'
graph.render('Decision Tree 1A')
DT1_prediction=DT1.predict(TestDF)
DT1_matrix=confusion_matrix(TestLabels,DT1_prediction)
print(DT1_matrix)
Acc1=classification_report(TestLabels,DT1_prediction,target_names=topics)
print(Acc1)
##Feature Importances
FeatureImp1=DT1.feature_importances_
indices=np.argsort(FeatureImp1)[::-1]
for f in range(TrainDF.shape[1]):
    if FeatureImp1[indices[f]]>0:
        print("%d. feature %d (%f)"%(f+1,indices[f],FeatureImp1[indices[f]]))
        print("feature name:",feature_names[indices[f]])
plt.barh(feature_names[indices[0:16]],FeatureImp1[indices[0:16]], 
         color ='blue')
plt.xlabel("Feauture Importances",fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel("Feature",fontsize=20)
plt.yticks(fontsize=15)
plt.title("Feature Importances DT1",fontsize=30)
plt.show()

##Decision Tree 2
DT2=DecisionTreeClassifier(criterion='entropy',min_samples_split=10,
                           splitter='best',ccp_alpha=.003)
DT2.fit(TrainDF,TrainLabels)
feature_names=TrainDF.columns
Tree_Object=tree.export_graphviz(DT2,out_file=None,feature_names=feature_names,
                                 class_names=list(set(Labels)),filled=True,rounded=True,
                                 special_characters=True)
graph=graphviz.Source(Tree_Object)
graph.format='png'
graph.render('Decision Tree 2C')
DT2_prediction=DT2.predict(TestDF)
DT2_matrix=confusion_matrix(TestLabels,DT2_prediction)
print(DT2_matrix)
Acc2=classification_report(TestLabels,DT2_prediction,target_names=topics)
print(Acc2)

##Feature Importances
FeatureImp2=DT2.feature_importances_
indices=np.argsort(FeatureImp2)[::-1]
for f in range(TrainDF.shape[1]):
    if FeatureImp2[indices[f]]>0:
        print("%d. feature %d (%f)"%(f+1,indices[f],FeatureImp2[indices[f]]))
        print("feature name:",feature_names[indices[f]])
plt.barh(feature_names[indices[0:20]],FeatureImp2[indices[0:20]], 
         color ='blue')
plt.xlabel("Feauture Importances",fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel("Feature",fontsize=20)
plt.yticks(fontsize=15)
plt.title("Feature Importances DT2",fontsize=30)
plt.show()


##Decision Tree 3
DT3=DecisionTreeClassifier(criterion='entropy',min_samples_split=15,
                           splitter='best',ccp_alpha=.006)
DT3.fit(TrainDF,TrainLabels)
feature_names=TrainDF.columns
Tree_Object=tree.export_graphviz(DT3,out_file=None,feature_names=feature_names,
                                 class_names=list(set(Labels)),filled=True,rounded=True,
                                 special_characters=True)
graph=graphviz.Source(Tree_Object)
graph.format='png'
graph.render('Decision Tree 3C')
DT3_prediction=DT3.predict(TestDF)
DT3_matrix=confusion_matrix(TestLabels,DT3_prediction)
print(DT3_matrix)
Acc3=classification_report(TestLabels,DT3_prediction,target_names=topics)
print(Acc3)

##Feature Importances
FeatureImp3=DT3.feature_importances_
indices=np.argsort(FeatureImp3)[::-1]
for f in range(TrainDF.shape[1]):
    if FeatureImp3[indices[f]]>0:
        print("%d. feature %d (%f)"%(f+1,indices[f],FeatureImp3[indices[f]]))
        print("feature name:",feature_names[indices[f]])
plt.barh(feature_names[indices[0:20]],FeatureImp3[indices[0:20]], 
         color ='blue')
plt.xlabel("Feauture Importances",fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel("Feature",fontsize=20)
plt.yticks(fontsize=15)
plt.title("Feature Importances DT3",fontsize=30)
plt.show()

