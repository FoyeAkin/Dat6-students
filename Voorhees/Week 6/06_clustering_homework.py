'''
CLUSTER ANALYSIS ON COUNTRIES
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.tools.plotting import parallel_coordinates

# Import the data
d = pd.read_csv('../data/UNdata.csv')
np.random.seed(0)
colors = np.array(['#FF0054','#FBD039','#23C2BC'])

# Run KMeans with k = 3
# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality

c=d[['GDPperCapita','lifeMale','lifeFemale','infantMortality']]

est = KMeans(n_clusters=3, init='random')
est.fit(c)
y_kmeans = est.predict(c)

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)

c['est'] =y_kmeans

plt.figure()
parallel_coordinates(c,'est', colors=('#FF0054','#FBD039','#23C2BC'))
   
#scatter
plt.figure(figsize=(8, 8))
plt.suptitle('Scatter Plot Grid',  fontsize=14)
# Upper Left
plt.subplot(221)
plt.scatter(c['lifeFemale'], c['GDPperCapita'], c = colors[y_kmeans])
plt.ylabel(c.columns[0])

# Upper Right
plt.subplot(222)
plt.scatter(c['infantMortality'], c['GDPperCapita'], c = colors[y_kmeans])

# Lower Left
plt.subplot(223)
plt.scatter(c['lifeFemale'], c['lifeMale'], c = colors[y_kmeans])
plt.ylabel(c.columns[1])
plt.xlabel(c.columns[2])

# Lower Right
plt.subplot(224)
plt.scatter(c['infantMortality'], c['lifeMale'], c = colors[y_kmeans])
plt.xlabel(c.columns[3])

# Print out the countries present within each cluster. Do you notice any general trend?
d['est']=y_kmeans
'''
x=pd.DataFrame(d.groupby('est').country.value_counts(),index=None)
x.reset_index(inplace=True)
'''
# Print out the properties of each cluster. What are the most striking differences?
d.groupby()             
d.sort(columns=['region', 'est'], inplace=True)

# Advanced: Re-run the cluster analysis after centering and scaling all four variables 

GDPperCapita= pd.DataFrame((c['GDPperCapita']-c['GDPperCapita'].mean()) / c['GDPperCapita'].std())
lifeMale= pd.DataFrame((c['lifeMale']-c['lifeMale'].mean()) / c['lifeMale'].std())
lifeFemale= pd.DataFrame((c['lifeFemale']-c['lifeFemale'].mean()) / c['lifeFemale'].std())
infantMortality= pd.DataFrame((c['infantMortality']-c['infantMortality'].mean()) / c['infantMortality'].std())

c_scaled=pd.concat([GDPperCapita,lifeMale,lifeFemale,infantMortality],axis=1)

est_scaled = KMeans(n_clusters=3, init='random')
est_scaled.fit(c_scaled)
y_kmeans_scaled = est_scaled.predict(c_scaled)

d['est_scaled']=y_kmeans_scaled
c['est_scaled']=y_kmeans_scaled
c_scaled['est_scaled']=y_kmeans_scaled               
# Advanced: How do the results change after they are centered and scaled? Why is this?

plt.figure(figsize=(8, 8))
plt.suptitle('Scatter Plot Grid',  fontsize=14)
# Upper Left
plt.subplot(221)
plt.scatter(c_scaled['lifeFemale'], c_scaled['GDPperCapita'], c = colors[y_kmeans_scaled])
plt.ylabel(c.columns[0])

# Upper Right
plt.subplot(222)
plt.scatter(c_scaled['infantMortality'], c_scaled['GDPperCapita'], c = colors[y_kmeans_scaled])

# Lower Left
plt.subplot(223)
plt.scatter(c_scaled['lifeFemale'], c_scaled['lifeMale'], c = colors[y_kmeans_scaled])
plt.ylabel(c.columns[1])
plt.xlabel(c.columns[2])

# Lower Right
plt.subplot(224)
plt.scatter(c_scaled['infantMortality'], c_scaled['lifeMale'], c = colors[y_kmeans_scaled])
plt.xlabel(c.columns[3])
 
 
plt.figure()
parallel_coordinates(c_scaled,'est_scaled', colors=('#FF0054','#FBD039','#23C2BC'))
            
d.sort(columns=['region', 'est_scaled'], inplace=True)

               