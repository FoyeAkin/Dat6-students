'''
CLASS: Introduction to scikit-learn with iris data
'''

import numpy as np
import matplotlib.pyplot as plt

# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()

###############################
###### Data Exploration #######
###############################

# create X (features) and y (response)
X, y = iris.data, iris.target

# inspect X
X
X.shape

# inspect y
y
y.shape

iris.feature_names
iris.target_names

# plot the flowers' petal width cs sepal width
colors = np.array(['#FF0054','#FBD039','#23C2BC'])
plt.figure()
plt.scatter(X[:, 3], X[:, 1], c=colors[y])
plt.xlabel(iris.feature_names[3])
plt.ylabel(iris.feature_names[1])

# plot the flowers' petal length cs sepal length
colors = np.array(['#FF0054','#FBD039','#23C2BC'])
plt.figure()
plt.scatter(X[:, 2], X[:, 0], c=colors[y])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[0])


###############################
### Prediction with sklearn ###
###############################

'''
scikit-learn is the go to python module for machine learning. It's great because it follows a standard set of rules


The general model is:
	model = sklearn.model
	model.fit(X, y)
         where X and y are your predictors and response!
	model.predict(incoming_data)

And it is almost always like this!
'''

# predict y with KNN
from sklearn.neighbors import KNeighborsClassifier  # import class
knn = KNeighborsClassifier(n_neighbors=1)           # instantiate the estimator
knn.fit(X, y)                                       # fit with data
'''
POP QUIZ:
Of X and y,
which is the response?
Which is the set of predictors?

what is X.shape?
what is y.shape?

'''
knn.predict([3, 5, 4, 2])                           # predict for a new observation
iris.target_names[knn.predict([3, 5, 4, 2])]
knn.predict([3, 5, 2, 2])

# predict for multiple observations at once
X_new = [[3, 5, 4, 2], [3, 5, 2, 2]]


knn.predict(X_new)

# try a different value of K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.predict(X_new)              # predictions
knn.predict_proba(X_new)        # predicted probabilities
knn.kneighbors([3, 5, 4, 2])    # distances to nearest neighbors (and identities)
np.sqrt(((X[106] - [3, 5, 4, 2])**2).sum()) # Euclidian distance calculation for nearest neighbor

# compute the accuracy for K=5 and K=1
# accuracy in this case literally means
# percent of correct predictions.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.score(X, y)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)

# 100% Hmm, does that seem fishy to anyone else?


