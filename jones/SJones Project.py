# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:35:51 2015

@author: shannonjones
"""

import pandas as pd 


data = pd.read_csv('testdata.csv')
data.dropna()
data.head(15)
data.describe()

data.Sentiment.hist()

data['label'] = data.Sentiment.map({0:0, 1:0, 2:0, 3:1, 4:1})
data.head(30)
data.dropna()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.Phrase, data.Sentiment, random_state=1)
X_train.shape
X_test.shape



from sklearn.feature_extraction.text import CountVectorizer
# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
# fit_transofmr accomplishses both the "fit" and "transform" funciton in one line
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm

# store feature names and examine them
train_features = vect.get_feature_names()
len(train_features)
train_features[:50]
train_features[-50:]

# convert train_dtm to a regular array
train_arr = train_dtm.toarray()
train_arr

import numpy as np

# exercise: calculate the number of tokens in the 0th message in train_arr
np.sum(train_arr[0,:])
# exercise: count how many times the 0th token appears across ALL messages in train_arr
np.sum(train_arr[:,0])
# exercise: count how many times EACH token appears across ALL messages in train_arr
np.sum(train_arr,axis=0)
# exercise: create a DataFrame of tokens with their counts
pd.DataFrame({'token':train_features, 'count':np.sum(train_arr,axis=0)})

# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
preds = nb.predict(test_dtm)
(preds)

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, preds)
print metrics.confusion_matrix(y_test, preds)

probs = nb.predict_proba(test_dtm)[:, 1]
probs
print metrics.roc_auc_score(y_test, probs)


#try random forest