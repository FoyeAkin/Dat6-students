'''
Homework: Analyzing the drinks data

    Drinks data
    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption

'''
import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
import operator 
# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10) 



# Examine the data types of all columns
drinks.dtypes


# Print the 'beer_servings' Series
drinks[['beer_servings']]
drinks.beer_servings  #equivalent print but without structured brackets around data
# Calculate the average 'beer_servings' for the entire dataset
drinks.beer_servings.mean()
average = int(drinks.beer_servings.sum()/len(drinks.beer_servings))
print("The average beer servings is:")
print(average)
# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent == 'EU']

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent == 'EU'].beer_servings.mean()



# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent == 'EU') & (drinks.wine_servings > 300)]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by=['total_litres_of_pure_alcohol']).tail(10)
# Determine which country has the highest value for 'beer_servings'
drinks.beer_servings.max()
drinks.sort_index(by=['beer_servings','country']).tail(1) #checking for answer. Namibia should be returned
drinks.sort_index(by=['beer_servings','country'],ascending=[False,True], inplace=True)

# Count the number of occurrences of each 'continent' value and see if it looks correct
#yeah, sure. 
# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()].country