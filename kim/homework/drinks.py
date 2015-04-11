# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:37:04 2015

@author: sunggyunkim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes

# Print the 'beer_servings' Series
drinks.beer_servings

# Calculate the average 'beer_servings' for the entire dataset
(drinks.beer_servings).sum()/drinks.shape[0]

# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent == 'EU'].shape

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent == 'EU'].beer_servings.sum()/drinks[drinks.continent == 'EU'].shape[0]

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent == 'EU') & (drinks.wine_servings > 300)]
# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol',ascending=[False]).head(10)

# Determine which country has the highest value for 'beer_servings'
drinks.sort_index(by="beer_servings", ascending=[False]).head(1)

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()]
