"""
Matt Reese
GA Data Sciences
Homework 3 - Due March 14, 2015

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

#Pull IMDB data
r = requests.get('http://bit.ly/cs109_imdb').text

#Create a list 'names' to categorize the data
names = ['imdbID','title','year','score','votes','runtime','genres']

#Create a dataframe using r's URL as a datasource, names as a header
#dropna removes any rows that have a blank
data = pd.read_csv('http://bit.ly/cs109_imdb', delimiter = '\t', names = names).dropna()
data.head()

#Cleaning Data

#Runtime is currently a string, but it should be an integer.

#This command gives the integer version of a single row's runtime.
#Now we need to make this iterate over the entire list
int(data['runtime'][0][:-6])
type(data.runtime[0])

#We use list comprehension
#Stolen from Sinan
data.runtime
int_times = [int(num[:-6]) for num in data.runtime]

#Code from online
dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

data['runtime'] = [float(num[:-6]) for num in data.runtime]
data['runtime']
data.head()

#Splitting up genres
#we use a set. Sets are unordered and can only have hashable values.
#They also cannot have duplicates.
#For each element in data.genres:
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

#make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]

#Remove the year number from the movie titles.
#We can use the same method of list comprehension we used above
data['title'] = [shortened_title[:-7] for shortened_title in data.title]
data['title'].head()

#Exploring global properties
data[['score','runtime','year','votes']].describe()

#A min runtime of 0? Weird. Let's see how many:
data.runtime[data.runtime==0]

#Let's remove them
data.runtime[data.runtime == 0] = np.nan

#Let's look at the overall data again
data[['score','runtime','year','votes']].describe()

#Plot some stuff

#Number of movies per year
data.year.hist(bins=62, color = 'cyan')
plt.xlabel('Release Date')

#Number of times movies achieved a specific score
data.score.hist(bins = 20, color = 'red')
plt.xlabel('Score')

#Number of movies with a specific runtime
data.runtime.dropna().hist(bins = np.arange(50,200), color = 'green')
plt.xlabel('Runtimes')

#Is there a correlation between year and score?
plt.scatter(data.year,data.score, lw = .5, alpha = .1)
plt.xlabel('Year')
plt.ylabel('Score')

#Let's look at score
plt.scatter(data.year,data.votes, lw = .5, alpha = .1, color = 'red')
plt.xlabel('Year')
plt.ylabel('Score')

#The scale is odd. Let's try using a logarithmic scale
plt.yscale('log')

#Let's look for movies with specific parameters
data[(data.votes > 9e4) & (data.score < 5)][['title','year','score','votes']]

data[data.score == data.score.max()][['title','year','score','votes','genres']]

#Sort by genre count
genre_count = np.sort(data[genres].sum())[::-1]
genre_count
genre_sums = data[genres].sum()
pd.DataFrame({'Genre Count':genre_sums}).sort(columns='Genre Count')[::-1]

genre_count2 = data[genres].sum(axis=1)
print 'The average movie has %2f genres' % genre_count2.mean()

#Split by decade
decade = (data.year // 10) * 10
tyd = data[['title','year']]
tyd['decade'] = decade

decade_mean = data.groupby(decade).score.mean()
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")

grouped_scores = data.groupby(decade).score
mean = grouped_scores.mean()
std = grouped_scores.std()

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)
plt.fill_between(decade_mean.index, (decade_mean + (2 * std)).values, (decade_mean - (2 * std)).values, color='g',alpha=.2)
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")

#Iterate over years
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values

#Homework:

#Question 1: Which genres are the most common?

#Figuring out datatypes
data.loc[1:]
type(data.genres)
type(data.genres[['Western']].mean())
type(data.genres)
type(data.genres[['Western']])

#Figuring out how to make a DataFrame
genre_means = pd.DataFrame()
index = [['Average']]
genre_means.set_index(index)

#Figuring out how to make source material for a dataframe
average_dict = {}
for genre in genres:
    average_dict[genre] = data[genre].mean()

#Making a dataframe    
average_dataframe = pd.DataFrame(average_dict, index = ['Average'])
#Sorting it
average_dataframe.sort(['Average'], ascending = True, inplace = True)
#The most common genre is Drama, followed by comedy

#Question 2:Given the two most prolific genres, how often do they coincide?
def is_both_genres(dframe):
    return (dframe['Drama'] & dframe['Comedy'])
    
#1521 of the films have both genres.
data.apply(is_both_genres, axis = 1).value_counts()
