# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:07:52 2021

@author: jstei
"""
# =============================================================================
# Appending 2020 salary data, salary cap data, and 2020 2k rankings data
# to the parent dataframe
# =============================================================================
##Import necessary libraries
import pandas as pd
##Reading in the various dataframes
Salaries=pd.read_csv('2020salaries.csv')
Rankings=pd.read_csv('Extra_2krankings.csv')
DF=pd.read_csv('CombinedDF_salary.csv')
SalaryCap=pd.read_csv('SalaryCap.csv')
##Dropping the Salaries column that simply repeats the salary data
Salaries=Salaries.drop('X4',axis=1)

##Converting the name column into a combined column with name and year to more easily match with the parent dataframe
for i in range(len(Rankings)):
    Rankings.loc[i,'combined']=Rankings.loc[i,'name']+str(Rankings.loc[i,'year'])
Salaries.X2=Salaries.X2+'2020'

##Removing extraneous information from the column holding year values
for i in range(len(SalaryCap)):
    SalaryCap.loc[i,'X1']=SalaryCap.loc[i,'X1'][:4]
##Dropping all rows with irrelevant years (pre 2012)
SalaryCap=SalaryCap.drop([i for i in range(0,28)])
##Resetting the column indices
SalaryCap=SalaryCap.reset_index()

# =============================================================================
# Using a series of nested for loops to transfer the information from the 3 dataframes
# to the parent dataset
# =============================================================================
for a in range(len(Salaries)):
    name=Salaries.loc[a,'X2']
    for i in range(len(DF)):
        name2=DF.loc[i,'combined']
        if name2==name:
            DF.loc[i,'Salary']=Salaries.loc[a,'X3']
            
for a in range(len(Rankings)):
    name=Rankings.loc[a,'combined']
    for i in range(len(DF)):
        name2=DF.loc[i,'combined']
        if name2==name:
            DF.loc[i,'2K_ratings']=Rankings.loc[a,'ranking']

for a in range(len(SalaryCap)):
    year=str(SalaryCap.loc[a,'X1'])
    for i in range(len(DF)):
        year2=str(DF.loc[i,'year'])
        if year2==year:
            DF.loc[i,'SalaryCap']=SalaryCap.loc[a,'X2']
#Writing the output to a csv file
DF.to_csv('DF_SalaryCap.csv',index=False)
