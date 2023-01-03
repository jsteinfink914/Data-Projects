# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:41:14 2021

@author: jstei
"""

import pandas as pd
import numpy as np

def unique(list1):
    x=np.array(list1)
    return np.unique(x)

DF=pd.read_csv('RawDF.csv')
labels=['Unnamed: 0','Unnamed: 0.1','jersey_number','reference']
DF=DF.drop(labels, axis=1)
DF.year=DF.year.astype('object')
DF['team_changes']=0
duplicates=[]
drop_index_values=[]
for a in range(len(DF)):
    name=DF.loc[a,'combined']
    values=DF[DF.combined==name].index.values
    values=[i for i in values]
    if len(values)>1:
        duplicates.append(name)
        drop_index_values.append(values)

duplicates=unique(duplicates)
drop_index_values=[i for sublist in drop_index_values for i in sublist]
drop_index_values=unique(drop_index_values)
keys=[i for i in DF.columns]
columns=dict.fromkeys(keys)
dataframes=[]
sum_columns=['games_played','games_started','offensive_rebounds','defensive_rebounds','tech_fouls','ejections','foulouts','double_doubles','triple_doubles']
for i in duplicates:
    data=[]
    values=DF[DF.combined==i].index.values
    values=[i for i in values]
    for column in columns:
        point=[]
        for value in values:
            point.append(DF.loc[value,column])
        if column=="games_played":
            weights=[]
            for i in range(len(point)):
                weights.append(point[i]/sum(point))
            data.append(sum(point))
        elif column=='team_changes':
            data.append(len(values))
        elif any(i==column for i in sum_columns):
            data.append(sum(point))
        else:
            if DF[column].dtype=='O':
                data.append(DF.loc[values[0],column])
            else:
                product=[]
                for num1,num2 in zip(weights,point):
                    product.append(num1*num2)
                data.append(sum(product))
    result=dict(zip(columns,data))
    individualDF=pd.DataFrame(result,index=[0])
    dataframes.append(individualDF)

DuplicateDF=pd.concat(dataframes,axis=0)

DF=DF.drop(drop_index_values,axis=0)
Duplicate_Free_DF=pd.concat([DF,DuplicateDF],axis=0)
DF.year=DF.year.astype('int64')
Duplicate_Free_DF.year=Duplicate_Free_DF.year.astype('int64')
Duplicate_Free_DF.sort_values(by=['year'])

DuplicateDF.to_csv('DuplicateDF.csv',index=None)
Duplicate_Free_DF.to_csv('Duplicate_Free_DF.csv',index=None)
    







# for a in range(len(DF)):
#     name=DF.loc[a,'id']
#     year=DF.loc[a,'year']
#     values=DF[DF.id==name].index.values
#     values=[i for i in values]
#     for i in values:
#         if DF.loc[i,'year']
#     if len(values)>1:
#         for i in range(len(values)-1):
#             if DF.loc[i,'year']==DF.loc[(i+1),'year']:
#                 try:
#                     values.pop(i)
#                 except:
#                     pass
#         if len(values)>1:
#             if DF.loc[values[0],'experience']!=DF.loc[values[1],'experience']:
#                 continue
#             else:
#                 for i in range(len(values)):
#                     DF.loc[values[i],'experience']=float(DF.loc[a,'experience'])-i