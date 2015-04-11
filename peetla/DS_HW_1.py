# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 20:08:58 2015

@author: chiranjeevi.peetla
"""

import pandas as pd  # this line imports python package called pandas
import numpy as np
import matplotlib.pylab as plt

# Read drinks.csv into a dataframe called "drinks"
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes
drinks.info()

# Print the 'beer_servings' Series
drinks.beer_servings
drinks['beer_servings']