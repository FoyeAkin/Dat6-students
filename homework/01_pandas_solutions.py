'''
SOLUTIONS: Working with drinks data
'''
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes
drinks.info()

# Print the 'beer_servings' Series
drinks.beer_servings
drinks['beer_servings']

# Calculate the average 'beer_servings' for the entire dataset
drinks.describe()                   # summarize all numeric columns
drinks.beer_servings.describe()     # summarize only the 'beer_servings' Series
drinks.beer_servings.mean()         # only calculate the mean

# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent=='EU']

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent=='EU'].beer_servings.mean()

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol').tail(10)

# Determine which country has the highest value for 'beer_servings'
drinks[drinks.beer_servings==drinks.beer_servings.max()].country
drinks[['country', 'beer_servings']].sort_index(by='beer_servings', ascending=False).head(5)

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()].country