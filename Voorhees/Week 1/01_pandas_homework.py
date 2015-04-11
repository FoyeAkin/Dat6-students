'''
Homework: Analyzing the drinks data

    Drinks data
    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption

'''

import pandas as pd  # This line imports  (already installed) python package

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
print "First ten rows of drinks.csv: "
print drinks.head(10)

# Examine the data types of all columns
print "Summary of the drinks.csv: "
print drinks.describe()

# Print the 'beer_servings' Series
print "Beer Servings:"
print drinks.beer_servings

# Calculate the average 'beer_servings' for the entire dataset
print "Global average for beer servings= "+str(drinks.beer_servings.mean())

# Print all columns, but only show rows where the country is in Europe
print "Data for European Countries:"
print drinks[drinks.continent=='EU']

# Calculate the average 'beer_servings' for all of Europe
print "Average number of beer servings for all of Europe = "+str(+drinks.beer_servings.mean(drinks[drinks.continent=='EU']))

# Only show European countries with 'wine_servings' greater than 300
print "These countries have wine servings greater than 300:"
print drinks.country[drinks.wine_servings>300]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol',inplace=True,ascending=False)  
print "Top 10 countries by alcohol consumption"
print drinks.country.iloc[10]

# Determine which country has the highest value for 'beer_servings'
drinks.sort_index(by='beer_servings',inplace=True,ascending=False)  
print drinks.country.iloc[0]+" has the highest beer servings of any country"
drinks.country[drinks.beer_servings==drinks.beer_servings.max()] #Alternative way to get this without sorting
# Count the number of occurrences of each 'continent' value and see if it looks correct
print "Occurences of continent values: "
print str(drinks.continent.value_counts())

# Determine which countries do not have continent designations
print "These countries do not have continent designations: "
print drinks.country[drinks.continent.isnull()]