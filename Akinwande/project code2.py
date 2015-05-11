# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import relevant packages for data 
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#Reading in all the data
#if needed add , low_memory=False to read statment
dat= pd.read_csv('pythonprojectdata.csv', sep=',', low_memory=False)

#import data from previous years 2009-2012
dat2= pd.read_csv('pythonprojectdata121.csv', sep=',', low_memory=False)

dat3= pd.read_csv('pythonprojectdata111.csv', sep=',', low_memory=False)

dat4= pd.read_csv('pythonprojectdata101.csv', sep=',', low_memory=False)


#Before appending datasets create a data frame  to check that all data sets have the same general structure
col_mns=['data1', 'data2']

data1=pd.Series(dat.columns)
data2=pd.Series(dat2.columns)
data3=pd.Series(dat3.columns)
data4=pd.Series(dat4.columns)

index=range(0,950)

datA=pd.DataFrame(index=index, columns=col_mns)
datA['data1']=data1
datA['data2']=data2

#check to see if data3 truly only have 938 columns
datA['data3']=data3
datA.shape
dat3.shape

#Set data on top of each other
data_list=[data1, data2, data3, data4]

set1=[]
set2=[]
set3=[]
set4=[]

set_list=[set1, set2, set3, set4]
for  i in range(4):
    set_list[i]=set(data_list[i])
        
set_list[0]

same_columns=set_list[0] & set_list[1] & set_list[2] & set_list[3] 
type(same_columns)
        
dat
dat.info
dat.columns.values
list(same_columns)

#Appending
dat_new=dat[list(same_columns)]

dats=[dat2, dat3, dat4]

for i in range(3):
    dat_new=dat_new.append(dats[i][list(same_columns)])
    
dat_new.columns
dat_new.head(10)


#Exploration of variables
dat_new.isnull().sum()
dat_new.YEARQ

#Looking to see the relationship between the age of a single offender and hate crimes.
#The few hate crimes present hate crimes are commited by those 21 and older. 
dat_new.groupby(['V4237']).V4526HA4.value_counts()

dat_new.V4526HA4.value_counts()

dat_new.V4399.value_counts()

#created a year variable
dat_new['Year']=dat_new.YEARQ.astype(int)

dat_new.columns

dat_new_second=pd.DataFrame()
dat_new_second=dat_new

dat_new_second.Year
#cleaning up the data. Removing invalid entries
to_delete=['(2) No', '(3) Dont know', '(8) Residue']

for item in range(3):
    dat_new_second=dat_new_second[dat_new_second.V4399 != to_delete[item]]

dat_new_second.V4526HA4.value_counts()


a=np.array(dat_new_second.groupby(['V4526HA4']).IDHH.value_counts())
Hate_Crime_Counts=pd.DataFrame(a.T)

dat_new_second.duplicated(['V4012','IDHH']).sum()
dat_new_second=dat_new_second.drop(['V4045'!='(1) yes'])
dat_new_second=dat_new_second.drop(dat_new_second.duplicated(['V4012'] ,['IDHH']==True))

dat_new_second=dat_new_second[dat_new_second.V4526HA4 != '(9) Out of universe']


dat_new_second.groupby('Year').V4526HA4.value_counts()

#drop residues in variables you will use. change I don't know to No. And choose only the incident reported to police. 

#Splitting the data by year 
Train=dat_new_second[(dat_new_second.Year==2010) | (dat_new_second.Year==2011)]
Test=dat_new_second[(dat_new_second.Year==2012) | (dat_new_second.Year==2013)]

#Changing strings to numbers and making my response variable binary

#one Did incident occur at worksite
rec=[name[1].replace('()',' ' ) for name in Train.V4484]
Train['V4484']=rec
Train['V4484']= Train['V4484'].astype(int)

#2 Police confirm incident was a hate crime
rec=[name[1].replace('()',' ' ) for name in Train.V4526HA4]
rec2=[name[0].replace('8 ','2') for name in rec]
rec3=[name[0].replace('3', '2') for name in rec2]
rec4=[name[0].replace('2', '0') for name in rec3]
Train['V4526HA4']=rec4
Train['V4526HA4']=Train['V4526HA4'].astype(int)

#3 Is the job located in city, rural, or suburban area
rec=[name[1].replace('()',' ' ) for name in Train.V4483]
Train['V4483']=rec
Train['V4483']=Train['V4483'].astype(int)

#4 Major activities happening that week
rec=[name[1].replace('()',' ' ) for name in Train.V4480]
Train['V4480']=rec
Train['V4480']=Train['V4480'].astype(int)

#5 Is this multiple offenders' only crime 
rec=[name[1].replace('()',' ' ) for name in Train.V4286]
Train['V4286']=rec
Train['V4286']=Train['V4286'].astype(int)

#6 Was there anything taken question with respondents' original repsonses
rec=[name[1].replace('()',' ' ) for name in Train.V4287]
Train['V4287']=rec
Train['V4287']=Train['V4287'].astype(int)

#7 was something taken question with responses allocated by interviewer 
rec=[name[1].replace('()',' ' ) for name in Train.V4288]
Train['V4288']=rec
Train['V4288']=Train['V4288'].astype(int)

#8 Is this the single offender's only crime against respondent or household
rec=[name[1].replace('()',' ' ) for name in Train.V4247]
Train['V4247']=rec
Train['V4247']=Train['V4247'].astype(int)

#9 number of offenders

Train.V4248.values

rec=[name[0].replace('()',' ' ) for name in Train.V4248]
rec2=[name[0].replace('(','0') for name in rec]
Train['V4248']=rec2
Train['V4248']=Train['V4248'].astype(int)

Train.V4248.value_counts()

#10 multiple offenders' sex
rec=[name[1].replace('()',' ' ) for name in Train.V4249]
Train['V4249']=rec
Train['V4249']=Train['V4249'].astype(int)

#11 multiple offenders only male or female
rec=[name[1].replace('()',' ' ) for name in Train.V4250]
Train['V4250']=rec
Train['V4250']=Train['V4250'].astype(int)

#12 multiple offenders: age of the youngest
rec=[name[1].replace('()',' ' ) for name in Train.V4251]
Train['V4251']=rec
Train['V4251']=Train['V4251'].astype(int)

#13 multiple offenders: age of the oldest
rec=[name[1].replace('()',' ' ) for name in Train.V4252]
Train['V4252']=rec
Train['V4252']=Train['V4252'].astype(int)

#14 Relationship to multiple offenders
rec=[name[1].replace('()',' ' ) for name in Train.V4264]
Train['V4264']=rec
Train['V4264']=Train['V4264'].astype(int)

#15 One or more than one offfenders
rec=[name[1].replace('()',' ' ) for name in Train.V4234]
Train['V4234']=rec
Train['V4234']=Train['V4234'].astype(int)

#16 single offender: sex
rec=[name[1].replace('()',' ' ) for name in Train.V4236]
Train['V4236']=rec
Train['V4236']=Train['V4236'].astype(int)

#17 single offender: age
rec=[name[1].replace('()',' ' ) for name in Train.V4237]
Train['V4237']=rec
Train['V4237']=Train['V4237'].astype(int)

#18 single offender: gang member
rec=[name[1].replace('()',' ' ) for name in Train.V4238]
Train['V4238']=rec
Train['V4238']=Train['V4238'].astype(int)

#19 single offender: stranger
rec=[name[1].replace('()',' ' ) for name in Train.V4241]
Train['V4241']=rec
Train['V4241']=Train['V4241'].astype(int)

#20 injuries suffered
rec=[name[1].replace('()',' ' ) for name in Train.V4110]
Train['V4110']=rec
Train['V4110']=Train['V4110'].astype(int)

#21 How was the respondent attacked
rec=[name[1].replace('()',' ' ) for name in Train.V4093]
Train['V4093']=rec
Train['V4093']=Train['V4093'].astype(int)

#22 how offender threatened or tried to attack
rec=[name[1].replace('()',' ' ) for name in Train.V4077]
Train['V4077']=rec
Train['V4077']=Train['V4077'].astype(int)

#23 offender hit or attacked original answer given by respondent
rec=[name[1].replace('()',' ' ) for name in Train.V4059]
Train['V4059']=rec
Train['V4059']=Train['V4059'].astype(int)

#24 offender hit or attacked response allocated by interviewer
rec=[name[1].replace('()',' ' ) for name in Train.V4060]
Train['V4060']=rec
Train['V4060']=Train['V4060'].astype(int)

#25 did respondent personally see a visitor
rec=[name[1].replace('()',' ' ) for name in Train.V4048]
Train['V4048']=rec
Train['V4048']=Train['V4048'].astype(int)

#26 respondent present original
rec=[name[1].replace('()',' ' ) for name in Train.V4044]
Train['V4044']=rec
Train['V4044']=Train['V4044'].astype(int)

#27 respondent present allocated
rec=[name[1].replace('()',' ' ) for name in Train.V4045]
Train['V4045']=rec
Train['V4045']=Train['V4045'].astype(int)

#28 about what time did incident occur
rec=[name[1].replace('()',' ' ) for name in Train.V4021B]
Train['V4021B']=rec
Train['V4021B']=Train['V4021B'].astype(int)

#29 In what city/town/village did incident occur
rec=[name[1].replace('()',' ' ) for name in Train.V4022]
Train['V4022']=rec
Train['V4022']=Train['V4022'].astype(int)

#30 same county and state as resident
rec=[name[1].replace('()',' ' ) for name in Train.V4023]
Train['V4023']=rec
Train['V4023']=Train['V4023'].astype(int)

#31 How many times incident occur in the past 6 months
rec=[name[0:5].replace('()',' ' ) for name in Train.V4016]
rec2=[name[0].replace('(', '0') for name in rec]
Train['V4016']=rec2
Train['V4016']=Train['V4016'].astype(int)

#32 How many incidents occurred
rec=[name[1].replace('()',' ' ) for name in Train.V4017]
Train['V4017']=rec
Train['V4017']=Train['V4017'].astype(int)

#33 Month incident occurred
rec=[name[:6].replace('( )','' ) for name in Train.V4014]
rec2=[name[0:4].replace('(0', '') for name in rec]
rec3=[name[0:4].replace('(', '') for name in rec2]
rec4=[name[0:2].replace(')', '') for name in rec3]
Train['V4014']=rec4
Train['V4014']=Train['V4014'].astype(int)

#Creating dataframe df
new_extract=['V4484', 'V4483', 'V4480', 'V4286', 'V4287', 'V4288', 'V4247', 
             'V4248', 'V4249', 'V4250', 'V4251', 'V4252', 'V4264', 'V4234', 
             'V4236', 'V4237', 'V4238', 'V4241', 'V4110', 'V4093', 'V4077',
             'V4059','V4060', 'V4048', 'V4044', 'V4045', 'V4021B', 'V4022',
             'V4023', 'V4016', 'V4017', 'V4014', 'V4526HA4']

len(new_extract)

new_ext=[Train.V4484, Train.V4483, Train.V4480, Train.V4286, Train.V4287, Train.V4288, Train.V4247, 
             Train.V4248, Train.V4249, Train.V4250, Train.V4251, Train.V4252, Train.V4264, Train.V4234, 
             Train.V4236, Train.V4237, Train.V4238, Train.V4241, Train.V4110, Train.V4093, Train.V4077,
             Train.V4059, Train.V4060, Train.V4048, Train.V4044, Train. V4045, Train.V4021B, Train.V4022,
             Train.V4023, Train.V4016, Train.V4017, Train.V4014, Train.V4526HA4]

df=pd.DataFrame()

for i in range(-1,33):
    df[new_extract[i]]=new_ext[i]
    



#This is a function for classifying crimes

col_ix = {col:index for index, col in enumerate(df.columns)}

def tag_crime(df):
    if df[col_ix['V4484']] > 1 and df[col_ix['V4483']] > 1 and df[col_ix['V4480']] > 1 and df[col_ix['V4286']] > 1 and df[col_ix['V4287']] > 1 and df[col_ix['V4288']] > 1 and df[col_ix['V4247']] > 1 and df[col_ix['V4248']] > 1 and [col_ix['V4249']] > 1 and df[col_ix['V4250']] > 1 and df[col_ix['V4251']] > 1 and df[col_ix['V4252']] > 1 and df[col_ix['V4264']] > 1 and df[col_ix['V4234']] > 1 and df[col_ix['V4236']] > 1 and df[col_ix['V4237']] > 1 and df[col_ix['V4238']] > 1 and df[col_ix['V4241']] > 1 and df[col_ix['V4110']] > 1 and df[col_ix['V4093']] > 1 and df[col_ix['V4077']] > 1 and df[col_ix['V4048']] > 1 and df[col_ix['V4021B']] > 1 :
        return 'Hate Crime'
    else:
        return 'Not Hate Crime'
        
tag_crime(df.loc[327,:])   

 
 #Create predictions for classifying crimes
#predicted 41 Hate crimes
predictions = np.array([tag_crime(row) for row in Test.values])
Test['Predictions']=predictions

#I get a result of 0.0 here; the function is terrible at predicting.
np.mean(predictions == Test.V4526HA4.values)

#Something more advanced

#Splitting the data
train, test = train_test_split(df,test_size=0.3, random_state=1)

#dataframe
train = pd.DataFrame(data=train, columns=df.columns)
test = pd.DataFrame(data=test, columns=df.columns)

ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=33)

#checking how the dataframe looks
df.head(5)


ctree.fit(train.drop('V4526HA4', axis=1), train['V4526HA4'])

features = df.columns.tolist()[1:]

with open("Hc.dot", 'w') as f:
    f = tree.export_graphviz(ctree, out_file=f, feature_names=features, close=True)

ctree.classes_

ctree.feature_importances_

pd.DataFrame(zip(features, ctree.feature_importances_)).sort_index(by=1, ascending=False)

preds = ctree.predict(test.drop('V4526HA4', axis=1))

metrics.accuracy_score(test['V4526HA4'], preds)

#The tree predicted all 16 non-hate crimes accurately, but it did not correctly classify either of  the two hate crimes in hte saample.
#The specificity is really high, but the sensitivity is really low. There is a great chance for a hate crime to be misbranded with this model.
pd.crosstab(test['V4526HA4'], preds, rownames=['actual'], colnames=['predicted'])

# Make predictions on the test set using predict_proba
probs = ctree.predict_proba(test.drop('V4526HA4', axis=1))[:,1]

# Calculate the AUC metric
#The AUC metric is 0.88888888888888884

metrics.roc_auc_score(test['V4526HA4'], probs)

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

#I'm going to improve my metrics, by narrowing the variables I used. 

#Police confirmation
y = df['V4526HA4'].values

#Narrowed metrics down to five variables:
""" number of offenders, relation to multiple offenders, 
occurred in indian reservation or american indian land, 
injuries suffered, incident occurred at work site"""
X = df[['V4248', 'V4264', 'V4021B', 'V4110', 'V4484']].values

ctree = tree.DecisionTreeClassifier(max_depth=3)
np.mean(cross_val_score(ctree, X, y, cv=5, scoring='roc_auc'))

#mean score ended up being 0.70363636363636362

ctree = tree.DecisionTreeClassifier(max_depth=10)
np.mean(cross_val_score(ctree, X, y, cv=5, scoring='roc_auc'))

# looking for the best tree depth
ctree = tree.DecisionTreeClassifier(random_state=1, min_samples_leaf=20)
depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(X, y)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

Pokemon_trainer = grid.best_estimator_

#When I get a chance to download graphviz I would love th see the the diagram.
with open("Hc_best.dot", 'w') as f:
    f = tree.export_graphviz(Pokemon_trainer, out_file=f, feature_names=features, close=True)

"""Final thoughts: The decision tree did a much better at classifying hate crimes than I am. 
Next time I shall include data such as race and whether the crime occurred during activities and events that are typically observed by certain group people observe.
This might improve notonly my tree, but also my function.  """