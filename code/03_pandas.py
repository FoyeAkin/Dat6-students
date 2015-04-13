'''
Advanced Data Analysis in Python

    ----Drinks data----
    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
    Article Analyzing the Dataset: http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/

    ----UFO data----
    Scraped from: http://www.nuforc.org/webreports.html
    Article Analyzing the Dataset: http://josiahjdavis.com/ 
    
'''

# Import pandas package
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

'''
Plotting

(Modified from excellent examples created by Kevin Markham)
'''

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# bar plot of number of countries in each continent
drinks.continent.value_counts().plot(kind='bar', title='Countries per Continent')
plt.xlabel('Continent')
plt.ylabel('Count')
plt.show()                                  # show plot window (if it doesn't automatically appear)
plt.savefig('countries_per_continent.png')  # save plot to file

# histogram of beer servings (shows the distribution of a numeric column)
drinks.beer_servings.hist(bins=20)
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')

# density plot of beer servings (smooth version of a histogram)
drinks.beer_servings.plot(kind='density', xlim=(0,500))
plt.xlabel('Beer Servings')

# grouped histogram of beer servings (shows the distribution for each group)
drinks.beer_servings.hist(by=drinks.continent)
drinks.beer_servings.hist(by=drinks.continent, sharex=True)
drinks.beer_servings.hist(by=drinks.continent, sharex=True, sharey=True)
drinks.beer_servings.hist(by=drinks.continent, layout=(1, 5))   # change layout (new in pandas 0.15.0)

# boxplot of beer servings by continent (shows five-number summary and outliers)
drinks.boxplot(column='beer_servings', by='continent')

# scatterplot of beer servings versus wine servings
drinks.plot(kind='scatter', x='beer_servings', y='wine_servings', alpha=0.3)

# same scatterplot, except point color varies by 'spirit_servings'
# note: must use 'c=drinks.spirit_servings' prior to pandas 0.15.0
drinks.plot(kind='scatter', x='beer_servings', y='wine_servings', c=drinks.spirit_servings, colormap='Blues')

# same scatterplot, except all European countries are colored red
colors = np.where(drinks.continent=='EU', 'r', 'b')
drinks.plot(x='beer_servings', y='wine_servings', kind='scatter', c=colors)

# scatterplot matrix of all numerical columns
pd.scatter_matrix(drinks)

'''
Indexing
'''

# Read in ufo data
ufo = pd.read_csv('../data/ufo.csv')   
ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=True)

# Create a new index
ufo.set_index('State', inplace=True)
ufo.head()
ufo.index
ufo.index.is_unique

ufo.set_index('City', inplace=True) # Replaces existing index
ufo.set_index('City', inplace=True, append=True) # Adds to existing index
ufo.sort_index(inplace=True)

# Slice using the index
ufo.loc['WY',:]
ufo.loc[['ND', 'WY'],:]
ufo.loc['ND':'WY',:]                # Error because of sorting
ufo.sort_index(inplace=True)
ufo.loc['ND':'WY',:]

# Reset the index
ufo.reset_index(level='City', inplace=True) # Remove a certain label from the index
ufo.reset_index(inplace=True)               # Remove all labels from the index

# Create a multi-index
ufo.set_index(['State', 'City'], inplace=True)
ufo.sort_index(inplace=True)
ufo.head()
ufo.index

# Slice using the multi-index
ufo.loc[['ND', 'WY'],:]
ufo.loc['ND':'WY',:]
ufo.loc[('ND', 'Bismarck'),:]
ufo.loc[('ND', 'Bismarck'):('ND','Casselton'),:]

'''
Analyzing across time
'''

# Reset the index
ufo.reset_index(inplace=True)

# Convert Time column to date-time format (defined in Pandas)
# transforming to time
# Formatting Time: https://docs.python.org/2/library/time.html#time.strftime
ufo.dtypes
ufo['Time'] = pd.to_datetime(ufo['Time'], format="%m/%d/%Y %H:%M")
ufo.dtypes

# Compute date range
ufo.Time.min()
ufo.Time.max()

# Slice using time
ufo[ufo.Time > pd.datetime(1995, 1, 1)]
ufo[(ufo.Time > pd.datetime(1995, 1, 1)) & (ufo.State =='TX')]

# Set the index to time
ufo.set_index('Time', inplace=True)
ufo.sort_index(inplace=True)
ufo.head()

# Access particular times/ranges
ufo.loc['1995',:]
ufo.loc['1995-01',:]
ufo.loc['1995-01-01',:]

# Access range of times/ranges
ufo.loc['1995':,:]
ufo.loc['1995':'1996',:]
ufo.loc['1995-12-01':'1996-01',:]

# Access elements of the timestamp
# Date Componenets: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#time-date-components
ufo.index.year
ufo.index.month
ufo.index.weekday
ufo.index.day
ufo.index.time
ufo.index.hour

# Create a new variable with time element
ufo['Year'] = ufo.index.year
ufo['Month'] = ufo.index.month
ufo['Day'] = ufo.index.day
ufo['Weekday'] = ufo.index.weekday
ufo['Hour'] = ufo.index.hour

'''
Split-Apply-Combine
'''

# for each year, calculate the count of sightings
ufo.groupby('Year').City.count()
ufo.Hour.value_counts()             # Same as before

# for each Shape, calculate the first sighting, last sighting, and range of sightings. 
ufo.groupby('Shape').Year.min()
ufo.groupby('Shape').Year.max()

# Specify the variable outside of the apply statement
ufo.groupby('Shape').Year.apply(lambda x: x.max())

# Specifiy the variable within the apply statement
ufo.groupby('Shape').apply(lambda x: x.Year.max() - x.Year.min())

# Specify a custom function to use in the apply statement
def get_max_year(df):
    try:
        return df.Year.max()
    except:
        return ''
        
ufo.groupby('Shape').apply(lambda x: get_max_year(x))

# split/combine can occur on multiple columns at the same time
# for each Weekday / Hour combination, determine first sighting
ufo.groupby(['Weekday','Hour']).City.count()

'''
Merging data
'''

# Read in population data

pop = pd.read_csv('../data/population.csv')
pop.head()

ufo.head()

# Merge the data together
ufo = pd.merge(ufo, pop, on='State', how = 'left')

# Specify keys if columns have different names
ufo = pd.merge(ufo, pop, left_on='State', right_on='State', how = 'left')

# Observe the new Population column
ufo.head()

# Check for values that didn't make it (length)
len(ufo[ufo.Population.isnull()])

# Check for values that didn't make it (values)
ufo[ufo.Population.isnull()]

# Change the records that didn't match up using np.where command
ufo['State'] = np.where(ufo['State'] == 'Fl', 'FL', ufo['State'])

# Alternatively, change the state using native python string functionality
ufo['State'] = ufo['State'].str.upper()

# Merge again, this time get all of the records
ufo = pd.merge(ufo, pop, on='State', how = 'left')

'''
Writing Data
'''
ufo.to_csv('ufo_new.csv')               # First column is an index
ufo.to_csv('ufo_new.csv', index=False)  # First column is no longer index

'''
Other useful features
'''

# map values to other values
ufo['Weekday'] = ufo.Weekday.map({  0:'Mon', 1:'Tue', 2:'Wed', 
                                    3:'Thu', 4:'Fri', 5:'Sat', 
                                    6:'Sun'})

# Pivot rows to columns
ufo.groupby(['Weekday','Hour']).City.count()
ufo.groupby(['Weekday','Hour']).City.count().unstack(0) # Make first row level a column
ufo.groupby(['Weekday','Hour']).City.count().unstack(1) # Make second row level a column

# randomly sample a DataFrame
idxs = np.random.rand(len(ufo)) < 0.66   # create a Series of booleans
train = ufo[idxs]                        # will contain about 66% of the rows
test = ufo[~idxs]                        # will contain the remaining rows

# replace all instances of a value (supports 'inplace=True' argument)
ufo.Shape.replace('DELTA', 'TRIANGLE')   # replace values in a Series
ufo.replace('PYRAMID', 'TRIANGLE')       # replace values throughout a DataFrame

'''
Fun Stuff
'''

# Plot the number of sightings over time
ufo.groupby('Year').City.count().plot(  kind='line', 
                                        color='r', 
                                        linewidth=2, 
                                        title='UFO Sightings by year')
# -----Analysis-----
# Clearly, Aliens love the X-Files (which came out in 1993).
# Aliens are a natural extension of the target demographic so it makes sense.



# Plot the number of sightings over the day of week and time of day
ufo.groupby(['Weekday','Hour']).City.count().unstack(0).plot(   kind='line', 
                                                                linewidth=2,
                                                                title='UFO Sightings by Time of Day')
# -----Analysis-----
# This may be speculative, but it appears that Aliens also enjoy Keggers.



# Plot the sightings in in July 2014
ufo[(ufo.Year == 2014) & (ufo.Month == 7)].groupby(['Day']).City.count().plot(  kind='bar',
                                                        color='b', 
                                                        title='UFO Sightings in July 2014')
                                                        
# Plot multiple plots on the same plot (plots neeed to be in column format)
ufo_fourth = ufo[(ufo.Year.isin([2011, 2012, 2013, 2014])) & (ufo.Month == 7)]
ufo_fourth.groupby(['Year', 'Day']).City.count().unstack(0).plot(   kind = 'bar',
                                                                    subplots=True,
                                                                    figsize=(7,9))
# -----Analysis-----
# Aliens are love the 4th of July. The White House is still standing. Therefore
# it follows that Aliens are just here for the party.