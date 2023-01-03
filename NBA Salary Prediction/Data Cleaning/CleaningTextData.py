# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:09:05 2021

@author: jstei
"""
# =============================================================================
# Using sklearns CountVectorizer to vecotirze twitter data from 
# searches related to NBA stardom
# =============================================================================
##Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re
import json
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import warnings
warnings.filterwarnings('ignore')

##Establishing path to the fiels containing the raw tweets
path="C:/Users/jstei/Desktop/ANLY_501/Twitter_text_files"

FileNameList=os.listdir(path)
##Looping through the directory and adding each file name and
##paths to empty lists
MyFileList=[]
FileNames=[]
for file in os.listdir(path):
    fullpath=path+'/'+file
    MyFileList.append(fullpath)
    FileNames.append(file)
    


##List of words to ignore
IgnoreThese=["and", "And", "AND","THIS", "This", "this", "for", "FOR", "For", 
              "THE", "The", "the", "is", "IS", "Is", "or", "OR", "Or", "will", 
              "Will", "WILL","tailoring", "ask", "Ask",'NBA','nba','star',
              'Star','MVP','mvp','player','Player','basketball','game','team',
              'year','Year','season',"Season",'legend','Legend','Star','star',
              'players']

##Looping through each file and tokenizing it, cleaning it
##and sorting into appropriate category (words, hashes, or links)

for file in FileNames:
    ##Empty list to contain links, words, and hashtags
    BagOfWords=[]
    BagOfHashes=[]
    BagOfLinks=[]
    Rawfilename=file
    with open("Twitter_text_files/"+file, 'r') as file:
        for line in file:
            tweetSplitter = TweetTokenizer(strip_handles=True, reduce_len=True)
            WordList=tweetSplitter.tokenize(line)
            ##Using regular expressions to isolate links and hashes
            regex1=re.compile('^#.+')
            regex2=re.compile('[^\W\d]')
            regex3=re.compile('^http*')
            regex4=re.compile('.+\..+')
            for item in WordList:
                if(len(item)>2):
                    if((re.match(regex1,item))):
                        ##These will be hashtags
                        ##remove the hashtag
                        newitem=item[1:] 
                        BagOfHashes.append(newitem)
                    elif(re.match(regex2,item)):
                        if(re.match(regex3,item) or re.match(regex4,item)):
                            BagOfLinks.append(item)
                        else:
                            BagOfWords.append(item)   
                    else:
                        pass
                else:
                    pass
    BigBag=BagOfWords+BagOfHashes
    R_FILE=open("Twitter/"+Rawfilename,"w")
    ###Look at the words and ignore my stopwords and write words to file
    for w in BigBag:
        if(w not in IgnoreThese):
            rawWord=w+" "
            R_FILE.write(rawWord)
    R_FILE.close()

##Using the newly created clean text files to make a DTM with CountVectorizer
path="C:/Users/jstei/Desktop/ANLY_501/Twitter"

FileNameList=os.listdir(path)

##Collecting files 
MyFileList=[]
FileNames=[]
for file in os.listdir(path):
    fullpath=path+'/'+file
    MyFileList.append(fullpath)
    FileNames.append(file)

##Using CountVectorizer to create a DF from corpus of tweets
MyCV=CountVectorizer(input='filename',stop_words='english',encoding='cp1252')
My_DTM=MyCV.fit_transform(MyFileList)
MyColNames=MyCV.get_feature_names()
print("The vocab is: ",MyColNames,'\n\n')
DF=pd.DataFrame(My_DTM.toarray(),columns=MyColNames)
print(DF)
print(FileNames)
##Assigning labels
CleanNames=[i for i in FileNames]
##Removing '.txt' from file names so the labels are just the file name 
Names=[]
for i in CleanNames:
    i=i.split('.')
    i=i[0]
    Names.append(i)
DF.insert(loc=0, column='LABEL', value=Names)
print(DF)

##Creating a Wordcloud for each of the documents
for file in FileNames:
    text = open(path+'/'+file).read()
    wordcloud = WordCloud().generate(text)
    # Open a plot of the generated image.
    #figure(figsize = (20,2))
    plt.figure(figsize=(50,40))
    plt.imshow(wordcloud)
               #, aspect="auto")
    plt.axis("off")
    ##trumpplt.show()
DF.to_csv('Twitter_Data.csv')


