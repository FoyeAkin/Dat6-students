'''
Homework: Analyzing the drinks data

    Drinks data
    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption

'''

#!/usr/bin/env python

import pandas as pd  # This line imports  (already installed) python package
import numpy as np
import matplotlib.pylab as plt
# Read drinks.csv into a DataFrame called 'drinks'

drinks = pd.read_csv('../data/drinks.csv')
drinks
drinks.sort_index(by=['country'], ascending=True, inplace=True)
# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes

# Print the 'beer_servings' Series
drinks.beer_servings

# Calculate the average 'beer_servings' for the entire dataset

drinks.beer_servings.mean()
#drinks.beer_servings.mean calculates the mean per observation

# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent== 'EU']

# Calculate the average 'beer_servings' for all of Europe
drinks.beer_servings[drinks.continent.isin(['EU'])].mean()
# Only show European countries with 'wine_servings' greater than 300
drinks[drinks.continent.isin(['EU']) & (drinks.wine_servings > 300)]
# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks[drinks.sort_index(by=['total_litres_of_pure_alcohol'], ascending=[False], inplace=True)]
drinks.head(10)
# Determine which country has the highest value for 'beer_servings'
drinks[drinks.beer_servings==drinks.beer_servings.max()]
# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts(dropna=False) #The North American Continent is missing.
#A check to see if the count percentages add up to 100.
((drinks.continent.value_counts(dropna=False)/drinks.shape[0])*100.00)
sum((drinks.continent.value_counts(dropna=False)/drinks.shape[0])*100.00) #Does not add up to exactly 100.
#A check to see how North American continents are categorized in the contenent index.
drinks[(drinks.country=='Mexico') | (drinks.country=='Canada') | (drinks.country=='USA')]
#All three North American countries have their continent missing.
# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()]

#Boxplot of the total litres of alcohol served on each continent
drinks.boxplot(column='total_litres_of_pure_alcohol', by='continent')
plt.xlabel('Continent')
plt.ylabel('Total Litres of Pure Alcohol')
plt.show()
plt.savefig('Total Litres of Pure Alcohol by Continent.png')

#Europe has the highest count of total litres of pure alcohol.
#South America has the lowest count of total litres of pure alcohol