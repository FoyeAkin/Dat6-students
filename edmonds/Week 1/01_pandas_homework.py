'''
Homework: Analyzing the drinks data

    Drinks data
    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption

'''
import pandas as pd

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes

# Print the 'beer_servings' Series
drinks['beer_servings']
drinks.beer_servings                         # Same thing.

# Calculate the average 'beer_servings' for the entire dataset
drinks['beer_servings'].sum() / float(drinks.shape[0])
drinks.beer_servings.mean()                 # Same thing.

# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent == 'EU']

# Calculate the average 'beer_servings' for all of Europe
drinks_EU = drinks[drinks.continent == 'EU']
drinks_EU['beer_servings'].sum() / float(drinks_EU.shape[0])
drinks_EU.beer_servings.mean()               # Same thing... Sigh.

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent == 'EU') & (300 < drinks.wine_servings)]
drinks_EU[300 < drinks_EU.wine_servings]     # Same thing.
drinks_EU[300 < drinks_EU.wine_servings].iloc[:,[0,3]] # Only relevant columns

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks_total = drinks.sort_index(by='total_litres_of_pure_alcohol',ascending=False)
drinks_total.head(10)
drinks_total.iloc[:10,[0,4]]                 # Same thing, w/ only relevant columns

# Determine which country has the highest value for 'beer_servings'
drinks_beer = drinks.sort_index(by='beer_servings',ascending=False)
drinks_beer.iloc[0,:2]

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()].country
