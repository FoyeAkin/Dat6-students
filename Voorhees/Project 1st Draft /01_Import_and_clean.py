# -*- coding: utf-8 -*-
'''
IMPORTANT: HIT F5 TO LOAD THE DATA QUICKLY AND RUN THIS WHOLE FILE
A VARIABLE CODEBOOK IS AVAILABLE IN "PROJECT WRITEUP.PDF"
'''


import pandas as pd
#read in the submission dataset for kaggle (has no demand information)
submit =  pd.read_csv('./raw_data/test.csv').dropna() 
submit.isnull().sum()==0 #clean!
#read in the data I'll use to fit the model. 
dat = pd.read_csv('./raw_data/train.csv').dropna() 
dat.isnull().sum()==0 #clean!

#Have to rename the count variable to "Demand" since count is a method in python
dat.rename(columns={'count':'demand'},inplace=True)


def FeatureEngineer(data):  
  
  #Break out granular time variable (format 2011-01-01 05:00:00)

  data.rename(columns={'season':'Season','workingday':'Workday'}, inplace=True)
  data['datetime_unformatted']=data.datetime
  data['datetime'] = pd.to_datetime(data.datetime)
  data.set_index('datetime', inplace=True)
  data['Year'] = data.index.year
  data['Month'] = data.index.month
  data['Day'] = data.index.day
  data['Weekday'] = data.index.weekday
  data['Hour'] = data.index.hour
  data.reset_index(inplace=True)
  
  #Now make some other features (groups) that're useful for graphing#
  data['AM_PM']= data.Hour  
  data.AM_PM[data.Hour<12]=0 #AM
  data.AM_PM[data.Hour>=12]=1 #PM

  data['Time_of_Day']= data.Hour  
  data.Time_of_Day[(data.Hour>=2) & (data.Hour<=5)]=0 #"Very Early AM" 
  data.Time_of_Day[(data.Hour>=6) & (data.Hour<=9)]=1 #"Morning Commute"
  data.Time_of_Day[(data.Hour>=10) & (data.Hour<=13)]=2 #"Mid-morning and lunch"
  data.Time_of_Day[(data.Hour>=14) & (data.Hour<=17)]=3 #"Mid Afternoon"
  data.Time_of_Day[(data.Hour>=18) & (data.Hour<=21)]=4 #"Evening Commute and HH rush"
  data.Time_of_Day[(data.Hour==22) | (data.Hour==23)| (data.Hour==0)| (data.Hour==1)]=5 #"Late PM"
  
  #Convert "feels like temperature" to farenheit 
  data['atemp2']=data.atemp
  data['atemp']=data.atemp*9/5+32 

  data['Weather_Temp']= data.atemp    
  data.Weather_Temp[data.atemp<50]=0
  data.Weather_Temp[(data.atemp>=50) & (data.atemp<65)]=1
  data.Weather_Temp[(data.atemp>=65) & (data.atemp<80)]=2
  data.Weather_Temp[(data.atemp>=80) & (data.atemp<95)]=3
  data.Weather_Temp[(data.atemp>=95) & (data.atemp<110)]=4
  data.Weather_Temp[data.atemp>=110]=5
  
#run the feature engineer over training data; make dummies for some time variables  
FeatureEngineer(dat)
for d in ['Year','Month','Weekday','weather']:
    dummy=pd.get_dummies(dat[d],prefix=d).iloc[:,1:]
    dat=pd.concat([dat,dummy],axis=1)

#run the feature engineer over training data; make dummies for some time variables
FeatureEngineer(submit)
for d in ['Year','Month','Weekday','weather']:
    dummy=pd.get_dummies(submit[d],prefix=d).iloc[:,1:]
    submit=pd.concat([submit,dummy],axis=1)
del(dummy, d)
