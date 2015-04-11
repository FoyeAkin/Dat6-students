#-------------------------------------------#
# DAT6 - CLASS 6 HOMEWORK
# CLUSTER ANALYSIS ON COUNTRIES
#-------------------------------------------#
# Created by: Lena Nguyen - March 29, 2015
#-------------------------------------------#

from sklearn.cluster import KMeans
from pandas.tools.plotting import parallel_coordinates
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the data
d = pd.read_csv('/Users/Zelda/Data Science/DAT6/data/UNdata.csv')
np.random.seed(0)

# initial look at data
d.describe()

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)

# Ccatter plot of all variables in data
pd.scatter_matrix(d)
## Seems to be a link between infant mortality and GDP per capita.
## Link between life expectancy and GDP per capita is not as clear
## Male and female life expectancy variables show a collinear relationship

# Parallel Coordinates Plot
d2 = d[['region', 'lifeMale', 'lifeFemale', 'infantMortality']]
plt.figure()
parallel_coordinates(d2, 'region')
plt.show()
## GDP per Capita is on a much larger scale so the parallel coordinates plot
## is not very informative

# Run KMeans with k = 3
# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality
cols = ['lifeMale','lifeFemale','infantMortality','GDPperCapita']
est = KMeans(n_clusters=3)
est.fit(d[cols])
y_kmeans = est.predict(d[cols])
d['cluster'] = y_kmeans

#-------------------------------------------
# VISUALIZATION OF CLUSTER CHARACTERISTICS
#-------------------------------------------

# Print out the countries present within each cluster. Do you notice any general trend?
for x in range(0, 3):
    print '===== Countries in Cluster %s =====' % (x)
    print d['country'][d['cluster'] == x]

## Cluster 0 - developed countries
## Cluster 1 - developing countries
## Cluster 2  - mostly middle income countries, although it has some fairly wealthy countries
## Might have to do with inequality.

# Print out the properties of each cluster. What are the most striking differences?
d.groupby('cluster').agg(np.mean)
d.groupby('cluster').agg([np.min, np.max])
d.groupby('cluster').describe()
## One cluster has significantly more obs than the other two
## Average GDP per capita and infant mortality rate differ starkly between the clusters

# Colors for the 3 clusters
colors = np.array(['#DC143C','#32CD32','#FFA500'])
# Cluster 0 = red; cluster 1 = Green; cluster 2 =  Orange

# Scatter Plot Grid
plt.figure(figsize=(8, 8))
plt.suptitle('Scatter Plot Grid',  fontsize=14)
# Upper Left
plt.subplot(221)
plt.scatter(d['lifeFemale'], d['lifeMale'], c=colors[d['cluster']], alpha=0.7)
plt.xlabel('Life Expectancy of Women')
plt.ylabel('Life Expectancy of Men')
# Upper Right
plt.subplot(222)
plt.scatter(d['lifeFemale'], d['GDPperCapita'], c=colors[d['cluster']], alpha=0.7)
plt.xlabel('Life Expectancy of Women')
plt.ylabel('GDP Per Capita')
# Lower Left
plt.subplot(223)
plt.scatter(d['lifeFemale'], d['infantMortality'], c=colors[d['cluster']], alpha=0.7)
plt.xlabel('Life Expectancy of Women')
plt.ylabel('Infant Mortality Rate')
# Lower Right
plt.subplot(224)
plt.scatter(d['infantMortality'], d['GDPperCapita'], c=colors[d['cluster']], alpha=0.7)
plt.xlabel('Infant Mortality Rate')
plt.ylabel('GDP Per Capita')

#----------------------------------#
# KMEANS - STANDARDIZED VARIABLES
#----------------------------------#

# Advanced: Re-run the cluster analysis after centering and scaling all four variables
## Find z-score for the numeric variables
## Z score is value minus mean, divded by the std
## Most common method to standardize variables
zd = d[['country', 'region']]

for col in cols:    # cols defined above
    col_zscore = 'z'+col
    zd[col_zscore] = stats.zscore(d[col])

# Run KMeans with k = 3
zcols = ['zlifeMale', 'zlifeFemale', 'zinfantMortality', 'zGDPperCapita']
est = KMeans(n_clusters=3)
est.fit(zd[zcols])
zy_kmeans_2 = est.predict(zd[zcols])
zd['cluster2'] = zy_kmeans_2

# Advanced: How do the results change after they are centered and scaled? Why is this?
zd.groupby('cluster2').agg(np.mean)
zd.groupby('cluster2').describe()
## GDP per capita has a much larger scale than the other variables so that
## affected the clustering

# Parallel coordinate plots
zd2 = zd[['zlifeMale', 'zlifeFemale', 'zinfantMortality', 'zGDPperCapita', 'cluster2']]
plt.figure()
parallel_coordinates(zd2, 'cluster2')
plt.show()
# Cluster 0 - Avg/above avg life expectancy, low infant mortality rate, GDP per capita fairly low
# Cluster 1 - Below avg life expetancy, high infant mortality rate, GDP per capita low similar to GDP per capita
# Cluster 2 - Higher than avg life expectancy, low infant mortality rate, high GDP per capita

# Scatter plot grid
plt.figure(figsize=(8, 8))
plt.suptitle('Scatter Plot Grid',  fontsize=14)
# Upper Left
plt.subplot(221)
plt.scatter(zd['zlifeFemale'], zd['zlifeMale'], c=colors[zd['cluster']], alpha=0.7)
plt.xlabel('Life Expectancy of Women - Z score')
plt.ylabel('Life Expectancy of Men - Z score')
# Upper Right
plt.subplot(222)
plt.scatter(zd['zlifeFemale'], zd['zGDPperCapita'], c=colors[zd['cluster']], alpha=0.7)
plt.xlabel('Life Expectancy of Women - Z score')
plt.ylabel('GDP Per Capita - Z score')
# Lower Left
plt.subplot(223)
plt.scatter(zd['zlifeFemale'], zd['zinfantMortality'], c=colors[zd['cluster']], alpha=0.7)
plt.xlabel('Life Expectancy of Women - Z score')
plt.ylabel('Infant Mortality Rate - Z score')
# Lower Right
plt.subplot(224)
plt.scatter(zd['zinfantMortality'], zd['zGDPperCapita'], c=colors[zd['cluster']], alpha=0.7)
plt.xlabel('Infant Mortality Rate - Z score')
plt.ylabel('GDP Per Capita - Z score')
