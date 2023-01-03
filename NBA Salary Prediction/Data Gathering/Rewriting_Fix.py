# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:24:15 2021

@author: jstei
"""

import pandas as pd
import re
DF=pd.read_csv('Fix.csv')
Salary_2020=pd.read_csv('2020salaries.csv')
Salary_2019=pd.read_csv('nba2019-2020salaries.csv')
Salary=pd.read_csv('salaries 2003-2019.csv')
Ratings_extra=pd.read_csv('Extra_2krankings.csv')
Ratings=pd.read_csv('nba_rankings_2014-2020.csv')
All_Nba=pd.read_csv('All_NBA.csv',encoding='cp1252')

##First salary
##Nested for loop to merge the dataframe data
for i in range(len(Salary)):
    name=Salary.loc[i,'Player']
    ##remove punctuation 
    name=re.sub("[.' ]",'',name).lower()
    ##make combined value with name and year
    year=Salary.loc[i,'Season']-1
    combined=name+str(year)
    for a in range(len(DF)):
        ##Create combined values for both first and last name
        ##and full name
        name2=DF.loc[a,'full_name']
        name2=re.sub("[.' ]",'',name2).lower()
        name3=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name3=re.sub("[.' ]",'',name3).lower()
        combined2=name2+str(DF.loc[a,'year'])
        combined3=name3+str(DF.loc[a,'year'])
        if combined2==combined or combined3==combined:
            DF.loc[a,'Salary']=Salary.loc[i,"Salary"]
            
          

for i in range(len(Salary_2019)):
    name=Salary_2019.loc[i,'player']
    ##remove punctuation 
    name=re.sub("[.' ]",'',name).lower()
    ##make combined value with name and year
    year=Salary_2019.loc[i,'season'][:4]
    combined=name+str(year)
    for a in range(len(DF)):
        name2=DF.loc[a,'full_name']
        name2=re.sub("[.' ]",'',name2).lower()
        name3=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name3=re.sub("[.' ]",'',name3).lower()
        combined2=name2+str(DF.loc[a,'year'])
        combined3=name3+str(DF.loc[a,'year'])
        if combined2==combined or combined3==combined:
            DF.loc[a,'Salary']=Salary_2019.loc[i,"salary"]
          


for i in range(len(Salary_2020)):
    name=Salary_2020.loc[i,'X2']
    ##remove punctuation 
    name=re.sub("[.' ]",'',name).lower()
    ##make combined value with name and year
    year='2020'
    combined=name+year
    for a in range(len(DF)):
        name2=DF.loc[a,'full_name']
        name2=re.sub("[.' ]",'',name2).lower()
        name3=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name3=re.sub("[.' ]",'',name3).lower()
        combined2=name2+str(DF.loc[a,'year'])
        combined3=name3+str(DF.loc[a,'year'])
        if combined2==combined or combined3==combined:
            DF.loc[a,'Salary']=Salary_2020.loc[i,"X3"]
        
      
##Check to see if it worked
sum(DF['Salary'].isna())


##Now for the 2K ratings
for i in range(len(Ratings)):
    name=Ratings.loc[i,'PLAYER']
    ##remove punctuation 
    name=re.sub("[.' ]",'',name).lower()
    ##make combined value with name and year
    year=Ratings.loc[i,'SEASON'][:4]
    combined=name+str(year)
    for a in range(len(DF)):
        name2=DF.loc[a,'full_name']
        name2=re.sub("[.' ]",'',name2).lower()
        name3=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name3=re.sub("[.' ]",'',name3).lower()
        combined2=name2+str(DF.loc[a,'year'])
        combined3=name3+str(DF.loc[a,'year'])
        if combined2==combined or combined3==combined:
            DF.loc[a,'Ratings_2k']=Ratings.loc[i,"rankings"]

for i in range(len(Ratings_extra)):
    name=Ratings_extra.loc[i,'name']
    ##remove punctuation 
    name=re.sub("[.' ]",'',name).lower()
    ##make combined value with name and year
    year=Ratings_extra.loc[i,'year']
    combined=name+str(year)
    for a in range(len(DF)):
        name2=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name2=re.sub("[.' ]",'',name2).lower()
        name3=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name3=re.sub("[.' ]",'',name3).lower()
        combined2=name2+str(DF.loc[a,'year'])
        combined3=name3+str(DF.loc[a,'year'])
        if combined2==combined or combined3==combined:
            DF.loc[a,'Ratings_2k']=Ratings_extra.loc[i,"ranking"]
      

sum(DF['Ratings_2k'].isna())

##Now for the ALL_NBA
for i in range(len(All_Nba)):
    name=All_Nba.loc[i,'namePlayer']
    ##remove punctuation 
    name=re.sub("[.' ]",'',name).lower()
    ##make combined value with name and year
    year=All_Nba.loc[i,'yearSeason']-1
    combined=name+str(year)
    for a in range(len(DF)):
        name2=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name2=re.sub("[.' ]",'',name2).lower()
        name3=str(DF.loc[a,'first_name'])+str(DF.loc[a,'last_name'])
        name3=re.sub("[.' ]",'',name3).lower()
        combined2=name2+str(DF.loc[a,'year'])
        combined3=name3+str(DF.loc[a,'year'])
        if combined2==combined or combined3==combined:
            DF.loc[a,'All_Nba']=str(All_Nba.loc[i,"isAllNBA"])
            DF.loc[a,'All_Nba_team']=int(All_Nba.loc[i,'numberAllNBATeam'])

sum(DF['All_Nba'].isna())

DF.to_csv('Fixed.csv',index=False)
