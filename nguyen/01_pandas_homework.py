#------------------------------------------------------------------#
# Homework: Analyzing the drinks data
#
#    Drinks data
#    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
#------------------------------------------------------------------#
# Last edited by: Lena Nguyen, February 24, 2015 
#------------------------------------------------------------------#

#----------------#
# IMPORT MODULES #
#----------------#

import pandas as pd

#-------------#
# IMPORT DATA #
#-------------#

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.columns
drinks.dtypes

#-- NOT PART OF HW, I WAS JUST CURIOUS
#-- Summary of the data
drinks.info()

# Print the 'beer_servings' series
drinks['beer_servings']

# Calculate the average 'beer_servings' for the entire dataset
drinks['beer_servings'].describe()

#-- mean is included in this whole summary stats. 
#-- It's good to see all that info but if we wants to see ONLY THE MEAN:
drinks['beer_servings'].describe()['mean']

# Print all columns, but only show rows where the country is in Europe
# -- EU = Europe in continent column
drinks[drinks.continent == 'EU']

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent == 'EU']['beer_servings'].describe()

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent == 'EU') & (drinks['wine_servings']>300)]
#-- Only 3 countries

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol', ascending=False, inplace=True)
drinks.head(10)

# Determine which country has the highest value for 'beer_servings'
drinks.sort_index(by='beer_servings', ascending=False, inplace=True)
drinks.head(1)

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.groupby('continent').country.count()
#-- Country count looks right for the continents currently present
#-- but where are the Central and North American countries?

# Determine which countries do not have continent designations
drinks.continent.isnull().sum()
#-- 23 missing values for continent

drinks[drinks.continent.isnull()]
#-- Missing values seem to countries in:
#-- Central America, North America, and Carribbean 


