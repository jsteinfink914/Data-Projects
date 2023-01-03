# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:43:55 2021

@author: jstei
"""

# =============================================================================
# This code takes the clean and normalized text data from the 
# previously performed twitter searches regarding NBA stardom 
# grouped into the three clusters(but the code can work for any
# labeled DF). The code creates wordclouds for each of the labels,
# creates and checks a train and test split using a 80/20 split,
# and then performs SVM analysis making a confusion matrix and
# identifying the most relevant terms
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
import random as rd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
    
    ## This list holds all three wordclouds I am building
    List_of_WC.append(wc)
    

##Plotting the wordclouds
NumTopics=len(set(Labels))
topics=list(set(Labels))
for i in range(NumTopics):
    fig=plt.figure(figsize=(25, 25))
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.title(str.upper(topics[i]),fontsize=50)

###SVM

##Train test split of 80/20
rd.seed(4)
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


##Tuning cost paramaters
tuned_parameters=[
    {"kernel": ['linear'],'C':[.01,1,10,100,1000],'verbose':[1],'gamma':['auto']},
    {"kernel": ['rbf'],'C':[.01,1,10,100,1000],'verbose':[1],'gamma':['auto']},
    {"kernel": ['poly'],'C':[.01,1,10,100,1000],'verbose':[1],'gamma':['auto']}
    ]

##Using GridSearchCV to do cost tuning
clf=GridSearchCV(SVC(),tuned_parameters,n_jobs=-1)
##Fitting the GridSearchCV
tune=clf.fit(TrainDF,Trainlabel)
##Creating dataframe with returned info
Results=pd.DataFrame(tune.cv_results_)
Results[['param_C','param_gamma','param_kernel','mean_test_score','rank_test_score']]


##For loop for each kernel collecting ideal cost and running SVM
## Plotting feature importances for linear svm 
for i in ['linear','rbf','poly']:
    ##Isolating kernel results
    kernelDF=Results[Results.param_kernel==i]
    relevantDF=kernelDF[['param_C','param_gamma','param_kernel','mean_test_score','rank_test_score']]
    print(relevantDF,end='\n\n\n')
    ##Finding best model
    best_model=min(kernelDF.rank_test_score)
    index=kernelDF[kernelDF.rank_test_score==best_model].index.tolist()
    index=index[0]
    ##Finding the cost using index of best model
    cost=kernelDF.loc[index,'param_C']
    
    ##Run SVM
    print('Running SVM -',i,'with cost',cost,end='\n\n\n')
    SVM=sklearn.svm.SVC(C=cost,kernel=i,verbose=True,gamma='auto')
    model=SVM.fit(TrainDF,Trainlabel)
    predict=SVM.predict(TestDF)
    
    ##Plotting confusion matrix using seaborn heatmap
    Acc=confusion_matrix(Testlabel,predict)
    fig=plt.figure()
    sns.heatmap(Acc,annot=True,fmt='g',annot_kws={'size':20})
    plt.xlabel('Predicted labels',fontsize=20)
    plt.ylabel('True labels',fontsize=20)
    title='Confusion Matrix - SVM - '+i+' Kernel'
    plt.title(title,fontsize=30)
    plt.xticks(ticks=[0.5,1.5,2.5],labels=topics,fontsize=15)
    plt.yticks(ticks=[0.5,1.5,2.5],labels=topics,fontsize=15)

    ##Looking at classification report
    Acc1=classification_report(Testlabel,predict)
    print('Accuracy for SVM -',i+':')
    print(Acc1)
    ##Feature importances for linear kernel
    if i =='linear':
        for a in range(len(topics)):
            coef = SVM.coef_[a].ravel()
            top_positive_coefficients = np.argsort(coef,axis=0)[-10:]
            top_negative_coefficients = np.argsort(coef,axis=0)[:10]
            top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
            # create plot
            plt.figure(figsize=(15, 5))
            colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
            plt.bar(  x=  np.arange(20)  , height=coef[top_coefficients], width=.5,  color=colors)
            feature_names = np.array(TrainDF.columns)
            plt.xticks(np.arange(0, 20), feature_names[top_coefficients], rotation=60,ha='right',fontsize=18)
            title='Feature Importances - SVM - '+str.upper(i)+' Kernel: '+topics[a]
            plt.title(title,fontsize=25)
            plt.show()