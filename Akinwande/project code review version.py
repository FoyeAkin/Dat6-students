# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#if needed add , low_memory=False to read statment
dat= pd.read_csv('C:\Users\costco\Desktop\DAT6-master\Akinwande\#.csv', sep=',', low_memory=False)

dat
dat.info
dat.columns.values

dat.head(10)

#Cleaning up variables I want to take a look at.
dat['V4526AA'].value_counts()
dat['V4526HA4'].value_counts()
dat['V4526HA3'].value_counts()
dat['V4526HA2'].value_counts()
dat['V4526HA1'].value_counts()

#creating and adding new cleaned variables to dataframe
rec_V4526AA= [name[1].replace('(',' ' ) for name in dat.V4526AA]
rec_V4526AA
dat['perceived_hate_crime']= rec_V4526AA
dat['perceived_hate_crime'].value_counts()

rec_V4526HA4= [name[1].replace('(',' ' ) for name in dat.V4526HA4]
rec_V4526HA4
dat['police_confirmation']=rec_V4526HA4
dat['police_confirmation'].value_counts()

rec_V4526HA3= [name[1].replace('(',' ' ) for name in dat.V4526HA3]
rec_V4526HA3
dat['offender_used_hate_symbols']=rec_V4526HA3
dat['offender_used_hate_symbols']

rec_V4526HA2= [name[1].replace('(',' ' ) for name in dat.V4526HA2]
rec_V4526HA2
dat['offender_used_abusive_language']=rec_V4526HA2
dat['offender_used_abusive_language']

rec_V4526HA1= [name[1].replace('(',' ' ) for name in dat.V4526HA1]
rec_V4526HA1
dat['victim_targeted_for_beliefs']=rec_V4526HA1
dat['victim_targeted_for_beliefs']

#YEAR=[name[4].replace('.',' ' ) for name in dat.YEARQ]

dat.columns.values

#created dataframe of the new variables and the year
df=dat[['YEARQ', 'perceived_hate_crime', 'police_confirmation', 'offender_used_hate_symbols', 'offender_used_abusive_language', 'victim_targeted_for_beliefs']]

dat['YEARQ'].value_counts()

df

#This is a function for classifying crimes
col_ix = {col:index for index, col in enumerate(df.columns)}
def tag_crime(data):
    if data[col_ix['perceived_hate_crime','offender_used_abusive_language','offender_used_hate_symbols','victim_targeted_for_beliefs']] == 1:
        return 'Hate Crime'
    else:
        return 'Random Crime'
        
 #Create predictions for classifying crimes
predictions = np.array([tag_crime(row) for row in df.values])
np.mean(predictions == df.police_confirmation.values)
#Predictions not loading maybe data has to be in a np array to begin with