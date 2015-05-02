
"""
@author: shannonjones
"""

import pandas as pd 



#full dataset contains Nan values. Need to understand how to remove.
data=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


data = data.dropna()
data.head(15)
test.head(15)
data.describe()

data.Sentiment.hist()

#if want to get rid of all the phrases
data['length']=[len(i) for i in data.Phrase]
data.head(20)
grouped=data.groupby(by=['SentenceId'])
data2=grouped.apply(lambda g: g[g['length'] == g['length'].max()])
data2.describe()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.Phrase, data.Sentiment, random_state=1)
X_train.shape
X_test.shape


from sklearn.feature_extraction.text import CountVectorizer
# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
# fit_transform accomplishses both the "fit" and "transform" funciton in one line
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
#DataFrame of tokens with their counts
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
#print metrics.roc_auc_score(y_test, probs) #doesn't work because not binary
X_test[y_test < preds]#false positives
X_test[y_test > preds] #false neagative

#try random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=3)
clf= clf.fit(train_arr, y_train)

test_arr=test_dtm.toarray()

preds2=clf.predict(test_arr)
(preds2)

metrics.accuracy_score(y_test, preds)


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', 
                 stop_words='english')
train_dtm = vect.fit_transform(X_train)
train_dtm														
