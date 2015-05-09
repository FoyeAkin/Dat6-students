# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:05:04 2015

@author: heatherhardway
"""
import pandas as pd
import time
import urllib2
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from ggplot import *
import holidays


dd= pd.read_csv("forcast_vs_actuals_2014.csv")

"""Convert strings to dates, separates month, day, year and weekday"""

dates = [time.strptime(dd.Date[i],'%m/%d/%Y %H:%M') for i in range(0,len(dd.Date))]
day = [dates[i].tm_mday for i in range(0,len(dates))]
month=  [str(dates[i].tm_mon) for i in range(0,len(dates))]
wday =  [dates[i].tm_wday for i in range(0,len(dates))]
year =  [dates[i].tm_year for i in range(0,len(dates))]
yday = [dates[i].tm_yday for i in range(0,len(dates))]



'''Add date columns to data frame'''

dd['day'] = ['0'*(2 - len(str(dd.day[i])))+str(dd.day[i]) for i in range(0,len(dd.Zip))]
dd['month'] = ['0'*(2 - len(str(dd.month[i])))+str(dd.month[i]) for i in range(0,len(dd.Zip))]
dd['wday'] = wday
dd['year'] = year
dd['yday'] = yday
dd['Zip'] = ['0'*(5 - len(str(dd.Zip[i])))+str(dd.Zip[i]) for i in range(0,len(dd.Zip))]

'''convert zip for those that start with a zero'''


'''Cluster locations based on longitude and latitude
Select only name and coordinates in dataframe to do clustering
'''
locales = dd[['NewName','Latitude','Longitude']]

est = KMeans(n_clusters=10, init='random')
est.fit(locales.iloc[:,1:3])
y_kmeans = est.predict(locales.iloc[:,1:3])
clusterc = pd.DataFrame(est.cluster_centers_,columns=['Latitude', 'Longitude'])
clusterc['predicted'] = range(0,10)

'''Add cluster prediced column to locations and original'''
locales['predicted'] = y_kmeans
dd['predicted'] = y_kmeans

'''find the closes actual data point to the center of each cluster -- addpen that zipcode'''
locales['dist_to_cluster']  = [(locales.Latitude[i]-clusterc.Latitude[locales.predicted[i]])**2 + (locales.Longitude[i]-clusterc.Longitude[locales.predicted[i]])**2 for i in range(0,len(locales.predicted))]
idlist = list(locales.groupby('predicted').dist_to_cluster.idxmin())
zips_clusters = list(dd.Zip[idlist])
clusterc['zips'] = zips_clusters

locales = locales.reset_index()
ggplot(locales, aes('Latitude', 'Longitude', colour='predicted'))+geom_point()

'''get weather for every day in days clus'''



def getweatherdata(qstring):
    f = urllib2.urlopen('https://api.weathersource.com/v1/d4804c82afaf1edd45c6/history_by_postal_code.json?period=day&postal_code_eq='+qstring+'fields=postal_code,country,timestamp,tempMax,tempAvg,tempMin,precip,snowfall,windSpdMax,windSpdAvg,windSpdMin,cldCvrMax,cldCvrAvg,cldCvrMin,dewPtMax,dewPtAvg,dewPtMin,feelsLikeMax,feelsLikeAvg,feelsLikeMin,relHumMax,relHumAvg,relHumMin,sfcPresMax,sfcPresAvg,sfcPresMin,spcHumMax,spcHumAvg,spcHumMin,wetBulbMax,wetBulbAvg,wetBulbMin')
    json_string = f.read()
    parsed_json = json.loads(json_string)
    return pd.DataFrame(parsed_json)



weatherdata = pd.DataFrame(columns=[u'cldCvrAvg', u'cldCvrMax', u'cldCvrMin', u'country', u'dewPtAvg', u'dewPtMax', u'dewPtMin', u'feelsLikeAvg', u'feelsLikeMax', u'feelsLikeMin', u'postal_code', u'precip', u'relHumAvg', u'relHumMax', u'relHumMin', u'sfcPresAvg', u'sfcPresMax', u'sfcPresMin', u'snowfall', u'spcHumAvg', u'spcHumMax', u'spcHumMin', u'tempAvg', u'tempMax', u'tempMin', u'timestamp', u'wetBulbAvg', u'wetBulbMax', u'wetBulbMin', u'windSpdAvg', u'windSpdMax', u'windSpdMin'])
count=0
for i in range(0,len(clusterc.predicted)):
    for j in range(0,15):
        offset = 25*j
        qstring  =days_clus.zips[i]+'&country_eq=US&timestamp_between=2014-01-01T00:00:00-05:00,2015-01-01T00:00:00-05:00&limit=25&offset='+str(offset)+'&'
        q = getweatherdata(qstring)
        weatherdata = weatherdata.append(q)
        print (i,j)
        time.sleep(10)
        count = count+1
        print count

weatherdata.to_csv('clustered_weather_data.csv')
