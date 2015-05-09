# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:49:13 2015

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


dd = pd.read_csv('updated_fva_data.csv')
weatherdata = pd.read_csv('clustered_weather_data.csv')

datesstrip = [date[0:10] for date in weatherdata.timestamp]    
weatherdata['formatedDate'] = datesstrip


wdates = [time.strptime(datesstrip[i],'%Y-%m-%d') for i in range(0,len(datesstrip))]
wyday = [wdates[i].tm_yday for i in range(0,len(wdates))]
weatherdata['yday'] = wyday

'''add holidays jewish/Stat -- add State/yday/forecast yesterdy/actuals yesterday, etc'''
holidays = [1,20,48,146,185,244,286,315,331,359]

dd['ratio'] = dd['actual']/dd['forecast']
dd['class_ratio'] = 0
dd.class_ratio[dd.ratio>1.5] = 1
dd.class_ratio[dd.ratio<0.5] = -1

temp = dd.merge(weatherdata, how="inner", left_on = ["formatedDate", "predicted","yday"], right_on = ["formatedDate", "predicted","yday"])
#temp = temp.dropna()

temp['yday_ystr'] = temp.yday-1 
temp['yday_tmmrw'] =temp.yday+1

dataystr = temp.copy()
datatmrw = temp.copy()

cols_tmrw = [x+'tmrw' for x in dataystr.columns]
cols_ystr = [x+'ystr' for x in datatmrw.columns]

dataystr.columns =  cols_ystr
datatmrw.columns = cols_tmrw

##pad missing values
temp.fillna(method='pad', inplace=True)

temp2 =dataystr.merge(temp,how="inner", left_on = ["NewNameystr","yday_ystrystr"], right_on = ["NewName", "yday"])
temp2.fillna(method='pad', inplace=True)
temp3 = datatmrw.merge(temp2, how="inner", left_on = ["NewNametmrw","yday_tmmrwtmrw"], right_on = ["NewName", "yday"])
temp3.fillna(method='pad', inplace=True)
temp3['days_to_holiday'] = [min(abs(temp3.yday[i]-holidays)) for i in range(0,len(temp3.yday)) ]



cols_all = [u'forecast', u'NewName',
 u'day', u'month', u'wday', u'yday',
 u'cldCvrAvg', u'cldCvrMax', u'cldCvrMin', u'dewPtAvg', u'dewPtMax', u'dewPtMin', u'feelsLikeAvg', 
 u'feelsLikeMax', u'feelsLikeMin', u'precip', u'relHumAvg', u'relHumMax', u'relHumMin', 
 u'sfcPresAvg', u'sfcPresMax', u'sfcPresMin', u'snowfall', u'spcHumAvg', u'spcHumMax', u'spcHumMin', 
 u'tempAvg', u'tempMax', u'tempMin', u'wetBulbAvg', u'wetBulbMax', u'wetBulbMin', u'windSpdAvg', u'windSpdMax', 
 u'windSpdMin', u'forecasttmrw', u'cldCvrAvgtmrw',
 u'cldCvrMaxtmrw', u'cldCvrMintmrw', u'dewPtAvgtmrw', u'dewPtMaxtmrw', u'dewPtMintmrw', u'feelsLikeAvgtmrw', 
 u'feelsLikeMaxtmrw', u'feelsLikeMintmrw', u'preciptmrw', u'relHumAvgtmrw', u'relHumMaxtmrw', 
 u'relHumMintmrw', u'sfcPresAvgtmrw', u'sfcPresMaxtmrw', u'sfcPresMintmrw', u'snowfalltmrw', u'spcHumAvgtmrw',
 u'spcHumMaxtmrw', u'spcHumMintmrw', u'tempAvgtmrw', u'tempMaxtmrw', u'tempMintmrw', u'wetBulbAvgtmrw', 
 u'wetBulbMaxtmrw', u'wetBulbMintmrw', u'windSpdAvgtmrw', u'windSpdMaxtmrw', u'windSpdMintmrw', 
 u'actualystr', u'forecastystr', u'ratioystr', u'class_ratioystr', u'cldCvrAvgystr', u'cldCvrMaxystr', u'cldCvrMinystr', u'dewPtAvgystr', u'dewPtMaxystr', 
 u'dewPtMinystr', u'feelsLikeAvgystr', u'feelsLikeMaxystr', u'feelsLikeMinystr', 
 u'precipystr', u'relHumAvgystr', u'relHumMaxystr', u'relHumMinystr', u'sfcPresAvgystr', u'sfcPresMaxystr', 
 u'sfcPresMinystr', u'snowfallystr', u'spcHumAvgystr', u'spcHumMaxystr', u'spcHumMinystr', u'tempAvgystr', 
 u'tempMaxystr', u'tempMinystr', u'wetBulbAvgystr', u'wetBulbMaxystr', u'wetBulbMinystr', u'windSpdAvgystr', 
 u'windSpdMaxystr', u'windSpdMinystr', 'days_to_holiday']

cols = [u'forecast', 
 u'day', u'month', u'wday', u'yday',
 u'cldCvrAvg',
 u'precip', u'snowfall', 
 u'tempAvg', u'windSpdAvg',  u'forecasttmrw', u'cldCvrAvgtmrw',
 u'preciptmrw', u'snowfalltmrw', u'tempAvgtmrw', u'windSpdAvgtmrw', 
 u'actualystr', u'forecastystr', u'cldCvrAvgystr', 
 u'precipystr',  u'snowfallystr', u'tempAvgystr', 
 u'windSpdAvgystr', u'days_to_holiday']
 


X = temp3[cols]
y = temp3['class_ratio']
yc = temp3['actual']


from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y)
ctree = tree.DecisionTreeClassifier(max_depth=6)


# Fit the decision tree classifier
ctree.fit(X_train, y_train)
ctree.score(X_test,y_test)
preds = ctree.predict(X_test)
np.mean(cross_val_score(ctree, X, yc, cv=5))



from sklearn.ensemble import RandomForestClassifier   
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

rforest = RandomForestClassifier(n_estimators=24, max_depth=7)
rforest.fit(X_train, y_train)
rforest.score(X_test, y_test)
np.mean(cross_val_score(rforest, X, y, cv=5))

cm = confusion_matrix(rforest.predict(X_test),y_test)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

nest = range(15,25)
param_grid = dict(n_estimators=nest)
grid = GridSearchCV(rforest, param_grid, cv=5)
grid.fit(X, y)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]


xtrees = ExtraTreesClassifier(n_estimators=24, max_depth=7)
xtrees.fit(X_train, y_train)
xtrees.score(X_test, y_test)
preds = xtrees.predict(X_test)
xxx = pd.DataFrame({"p":preds, "y":y_test})
xxx.groupby(['p','y']).p.count()



est = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4).fit(X_train, y_train)
est.score(X_test, y_test)      
preds = est.predict(X_test)
xxx = pd.DataFrame({"p":preds, "y":y_test})
xxx.groupby(['p','y']).p.count()


# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_




