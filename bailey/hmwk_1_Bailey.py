# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 10:02:42 2015

@author: Tom
"""

import pandas as pd

# Read drinks.csv into a DataFrame called 'drinks'
dfDrinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
dfDrinks.head(10)

# Examine the data types of all columns
dfDrinks.dtypes

# Print the 'beer_servings' Series
dfDrinks['beer_servings']

# Calculate the average 'beer_servings' for the entire dataset
dfDrinks['beer_servings'].mean()

# Print all columns, but only show rows where the country is in Europe
dfDrinks[dfDrinks.continent == 'EU']

# Calculate the average 'beer_servings' for all of Europe
dfDrinksEur = dfDrinks[dfDrinks.continent == 'EU']
dfDrinksEur['beer_servings'].mean()

# Only show European countries with 'wine_servings' greater than 300
dfDrinksEur[dfDrinksEur.wine_servings > 300]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
dfDrinks.sort_index(by='total_litres_of_pure_alcohol', ascending = False).head(10) 

# Determine which country has the highest value for 'beer_servings'
dfDrinks.sort_index(by='beer_servings', ascending = False).head(1) 

# Count the number of occurrences of each 'continent' value and see if it looks correct
dfDrinks.continent.value_counts()

# Determine which countries do not have continent designations
dfDrinks[pd.isnull(dfDrinks.continent)]