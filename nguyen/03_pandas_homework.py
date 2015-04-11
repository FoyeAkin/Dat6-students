#------------------------------------------------------------------------------------#
# DATA WRANGLING/VISUALIZATION                      
# Uses IMDB data as an example                      
# Data Source: 
# http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb
#
# NOTE: Homework is at the bottom starting from line 203
#
#------------------------------------------------------------------------------------#
# LAST EDITED BY: Lena Nguyen - March 12, 2015
#------------------------------------------------------------------------------------#

#-----------------#
# IMPORT MODULES  #
#-----------------#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#-----------------------------#
# IMPORT DATA INTO DATA FRAME #
#-----------------------------#

# URL where data is at
data_url = 'http://bit.ly/cs109_imdb'

# List of column labels
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv(data_url, delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]

del data_url

data.head()  # print the first 5 rows

#-------------------------#
# CLEANING THE DATA FRAME #
#-------------------------#

# Since the run time is reported only in minutes,
# Strip the run time to only contain the number of minutes 
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime] 
#why use float instead of int? - to be safe?
data['runtime'] = clean_runtime
data.head()

# Splitting up the genres into separate columns w/ boolean values
# For easier sorting and grouping later

# Determine the unique genres
# Create a set of all the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

# make a column for each genre using list comprehension
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

# Strip the year from the title
# Year already has its own column in the data

# (Year) is 6 characters and 7 includes the space after the name
data['title'] = [t[0:-7] for t in data.title]
data.head()

#------------------#
# DATA EXPLORATION #
#------------------#

data[['score', 'runtime', 'year', 'votes']].describe()

# hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

# probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

data.runtime.describe()

#-- BASIC PLOTS --#

# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)
#-- histogram of number of movies in data from certain years
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")

#-- histogram of number of movies with certain ratings
plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")

#-- histogram of run time of all the movies, ignores missing values
plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")

#-- Scatter plot of IMDB score against year released
#hmm, more bad, recent movies. Real, or a selection bias?
plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")

#-- Scatter plot of number of votes versus IMDB score
#-- Scale is in log form for easier visualization
plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')

#-------------------#
# LOOK AT OUTLIERS  #
#-------------------#

# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]
#-- I'm surprised there are only two movies that everyone agreed is bad

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]
#-- 7 movies tied for worse at rating of 1.5

#-- These movies have significantly fewer number of votes than the two movies
#-- with low scores but large number of votes. Why are people more compelled
#-- to rate those other two movies more? Those two movie were bigger blockbuster
#-- movies

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

#---------------------------#
# RUN AGGREGATION FUNCTIONS #
#---------------------------#

#-- Total number of movies in the different genres
# sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1] # last part reverses the sort to be in descending order
pd.DataFrame({'Genre Count': genre_count})
#-- ?? Why does the genre name not print out like in the example?

#axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

#---------------------------#
# EXPLORE GROUP PROPERTIES  #
#---------------------------#

decade =  (data.year // 10) * 10

tyd = data[['title', 'year']]
tyd['decade'] = decade
data['decade']=decade

tyd.head()


#mean score for all movies in each decade

decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

#-- Plot decade average line into previous scatter plot of all the ratings

#-- code for mean line
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
        
#-- code for scatter plot
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
#-- ?? How does python know to plot the two graphs together?

#-- include a standard deviation area to go with the average line
grouped_scores = data.groupby(decade).score

mean = grouped_scores.mean()
std = grouped_scores.std()

#-- code for average line below
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
#-- code for standard deviation area
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)  
#-- code for scatter plot
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
#-- Axis labels
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

#-- Finding the highest rated in each year by iterating over a groupby object
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values 

#--------------#
#-- HOMEWORK --#
#--------------#

# Look at movies with the least number of ratings
data[data.votes == data.votes.min()][['title', 'year', 'score', 'votes', 'genres']]
#--> It is interesting the the least number of votes is 1,356 votes.
#--> Seems like a very arbitrary number. 
#--> Was that an artifact of dropping certain obs to make dataset smaller?

# Movie with good IMDB ratings but not that many votes (under 10k votes)
data[(data.votes < 10000 ) & (data.score > 8.5)][['title', 'year', 'score', 'votes', 'genres']]
#--> All three movies are foreign films. The first two are Turksih films.

#----------#
# HW PLOTS #
#----------#

# Plot scores by average number of votes. Do people tend to vote more for movies they like?
data.groupby('score').votes.mean().plot( kind='line', color='r', 
                                        linewidth=2, 
                                        title='Score by Average Number of Votes')
plt.savefig('scores_by_votes.png')
#--> It does appear that people tend to vote more for movies they like. 
#--> The increase around 8 is exponential

# Plot score versus average number of votes for each decade
data.groupby(['decade', 'score']).title.count().unstack(0).plot(    kind='line', 
                                                                    linewidth=2, 
                                                                    title='Number of movies with certain scores by decade')
plt.savefig('scores_by_decade.png')
#--> The 2000s had a lot of movies. It looks like the distribution of 
#--> movies scores about the same in every decade.                 

# Mean of drama movies against mean of all movies by decades
drama = data[data.Drama==True]

drama.sort('score', ascending=False, inplace=True)
drama[['title', 'year', 'score', 'votes', 'genres']].head(5)
drama[['title', 'year', 'score', 'votes', 'genres']].tail(5)

drama_mean = drama.groupby(decade).score.mean()
drama_mean.name = 'Drama movies mean'
print drama_mean

# Code for lines for all movies and drama movies
plt.plot(drama_mean.index, drama_mean.values, 'o-',
        color='c', lw=3, label='Drama Movies Decade Average')
        
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='All Movies Decade Average')     
# code for scatter plot
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
plt.savefig('dramas_decades.png')
#--> LN: Drama movies have slightly better ratings 
#--> but the difference is very slight. Probably not significant. 
