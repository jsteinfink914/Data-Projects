# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:58:32 2021

@author: jstei
"""
# =============================================================================
# From kaggle, salaries all the way up to 2019 were available, 
# but 2020 salary data was not, as a result it had to be scraped from
# https://hoopshype.com/nba2k/2021-2022/
# =============================================================================
##Importing necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate

##Web scraping hoopshype.com for the required years of data
years=['2021-2022','2014-2015','2013-2014']
datasets=[]
for i in years:
    year=i
    url='https://hoopshype.com/nba2k/'+i+'/'
    page=requests.get(url)
    html=page.text
    ##Using BeautifulSoup to sort through the html text
    soup=BeautifulSoup(page.content,'lxml')
    soup.prettify()
    ##Finding the first table in the html page 
    ##(the one that holds the salaries)
    table=soup.find_all('table')[0]
    ##Converting the table into a pandas list
    tablestring=pd.read_html(str(table))
    ##Converting the pandas list into a tabulated string 
    ##which makes processing easier
    tabulated=tabulate(tablestring[0])
    ##Splitting the string by the newline character
    split=tabulated.split('\n')
    ##Creating an empty list
    data=[]
    ##Using a for loop to split each line into individual 
    ##data points (Name and rating)
    for i in split:
        i=i.split(' ')
        ##appending all non-empty characters into a list 
        ##whih will represent individual rows
        i[:]=[a for a in i if a!='']
        ##Appending each row into the dataframe
        data.append(i)
    ##Removing the first row which was a collection of dashes
    data.pop(0)
    ##Creating a dataframe from the list of rows
    NBA2k=pd.DataFrame(data=data)
    ##Dropping unnecessary columns of rank order,an index column and a column full of None values
    NBA2k=NBA2k.drop(5,axis=1)
    NBA2k=NBA2k.drop(0,axis=1)
    NBA2k=NBA2k.drop(1,axis=1)
    ##Combining first and last name columns into one full name column
    NBA2k['name']=NBA2k[2]+' '+NBA2k[3]
    ##Assigning column full of rank values to a column with an appropriate title
    NBA2k['ranking']=NBA2k[4]
    ##Dropping first name, last name and the non-titled rank columns
    NBA2k=NBA2k.drop(2,axis=1)
    NBA2k=NBA2k.drop(3,axis=1)
    NBA2k=NBA2k.drop(4,axis=1)
    NBA2k['year']=year[:4]
    ##Adjusting the year to reflect the start of the season
    for i in range(len(NBA2k)):
        NBA2k.loc[i,'year']=int(NBA2k.loc[i,'year'])-1
    ##Appending dataframes to a large list
    datasets.append(NBA2k)
##Creating a large dataframe from the subset of dataframes
Rankings=pd.concat(datasets,axis=0)
##Writing rankings to a csv file
Rankings.to_csv('Extra_2krankings.csv',index=False)
    