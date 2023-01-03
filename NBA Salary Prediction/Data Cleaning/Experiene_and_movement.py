# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 20:22:46 2021

@author: jstei
"""

import pandas as pd
DF=pd.read_csv('Duplicate_Free_DF.csv')
len(unique(DF.id))
DF1=pd.read_csv('RawDF.csv')
len(unique(DF.id))
