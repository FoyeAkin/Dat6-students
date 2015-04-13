'''
CLUSTER ANALYSIS ON COUNTRIES
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the data
d = pd.read_csv('../data/UNdata.csv')
np.random.seed(0)

# Run KMeans with k = 3
# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality
j_clusts=d[['GDPperCapita', 'lifeMale','lifeFemale', 'infantMortality']]
est = KMeans(n_clusters=3, init='random')
est.fit(j_clusts)
y_kmeans = est.predict(j_clusts)

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)
colors = np.array(['#FF0054','#FBD039','#23C2BC'])
plt.figure()
plt.scatter(d[0, 6], d[:, 0], c=colors[y_kmeans], s=50)

plt.xlabel()
plt.ylabel()

# Print out the countries present within each cluster. Do you notice any general trend?

#created a column of the y_kmeans
j_clusts
d['cluster']=y_kmeans
d.sort_index(by='cluster')
#tried to get the country based on cluster column I created, but it does not look right
d.country[d.cluster].sort_index('cluster')
d.country[d.cluster|d.GDPperCapita & d.lifeMale & d.lifeFemale & d.infantMortality].sort_index('cluster')
d.groupby('cluster').country
d.country.groupby('cluster')

#tried a function to call a cluster and associating countries, but it did not work
def get_country(a):
    if d.cluster==a.item():
        return d.country
get_country(1)
    
# Print out the properties of each cluster. What are the most striking differences?
             
# Advanced: Re-run the cluster analysis after centering and scaling all four variables 
             
# Advanced: How do the results change after they are centered and scaled? Why is this?