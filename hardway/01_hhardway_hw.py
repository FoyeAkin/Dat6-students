# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 19:17:27 2015

@author: heatherhardway
HW #1
"""
import pandas as pd
import numpy

drinks = pd.read_csv('data/drinks.csv')  
drinks.head(10)
drinks.dtypes
numpy.mean(drinks.beer_servings)


print(drinks[drinks.continent=='EU'])
numpy.mean(drinks.beer_servings[drinks.continent=='EU'])
print(drinks[(drinks.continent=='EU')&(drinks.wine_servings>300)])

print(drinks.sort('total_litres_of_pure_alcohol', ascending=False).country.head(10))
print(drinks.sort('beer_servings', ascending=False).country.head(10))

drinks.groupby('continent').count().country
drinks.country[drinks.continent.isnull()] 
drinks.continent.fillna(value='NA', inplace=True)
