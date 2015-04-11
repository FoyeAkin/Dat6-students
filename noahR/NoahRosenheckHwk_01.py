'''
Noah Rosenheck
Class 1 Homework: Pandas

   Check out this excellent example of [data wrangling and exploration in Pandas](http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb).

* Assignment:
    A1 Read through the entire IPython Notebook.
    A2 As you get to each code block, **copy it into your own Python script** and run the code yourself. Try to understand exactly how each line works. You will run into Python functions that you haven't seen before!
    A3 Explore the data on your own using Pandas. At the bottom of your script, write out (as comments) **two interesting facts** that you learned about the data, and show the code you used to find those facts.
    A4 Create **two new plots** that show something interesting about the data, and save those plots as files. Include the plotting code at the bottom of your script.
    A5 Add your **Python script and image files** to your folder of DAT6-students and create a pull request.

    *  "Don't worry about running (or trying to understand) the code in the 'Small Multiples' section."
'''


'''
The basic workflow is as follows:
    1 Build a DataFrame from the data (ideally, put all data in this object)
    2 Clean the DataFrame. It should have the following properties:
        - Each row describes a single object
        - Each column describes a property of that object
        - Columns are numeric whenever appropriate
        - Columns contain atomic properties that cannot be further decomposed
    3 Explore global properties. Use histograms, scatter plots, and aggregation functions to summarize the data.
    4 Explore group properties. Use groupby and small multiples to compare subsets of the data.*
'''
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# clean up the table display a little bit
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

# 1. Build a DataFrame:
!head imdb_top_10000.txt
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('imdb_top_10000.txt', delimiter = '\t', names = names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()

# 2. Clean the DataFrame:

# 2a. Fix the runtime column
'''
Example: extract the numbers from strings, and turn them into integers

dirty = '142 min.'
number, text = dirty.split(' ') # splits dirty at the (space), and defines 'number' as the string before the break and 'text' as the string after the break
clean = int(number) # converts a string of numbers into an integer
#print text
print number
'''

clean_runtime = [float(r.split(' ')[0]) for r in data.runtime] # creates the temporary directory "clean_runtimes" of runtimes with the numeric strings converted to floats
data['runtime'] = clean_runtime                                # sets the entires in 'runtime' to the entires in 'clean_runtimes'
data.head()

# 2b. Split the genres up

# determine the unique genres
genres = set()                             # creates a new empty set
for m in data.genres:
    genres.update(g for g in m.split('|')) # for each entry 'm' in data.genres, update the set with a list of strings of the characters between the ' | '
genres = sorted(genres)                    # writes the set into the column data.genres

# make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
data.head()

# 2c. Remove the year from the title
data['title'] = [t[0:-7] for t in data.title]
data.head()

# 3. Explore Global Properties!

# 3a. Call describe on relevant columns
data[['score', 'runtime', 'year', 'votes']].describe()


# 3b. count all entries with runtime 0, flag them as NaN, and call describe again
print len(data[data.runtime == 0])
data.runtime[data.runtime == 0] = np.nan
data.runtime.describe()

# 3c. Basic plots
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("release Year")

plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")

plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")

plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")

plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')

# 3d. Identify outliers
#i.   low score ovies with lots of votes
data[(data.votes > 9e4)& (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

#ii.  The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

#iii.The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

# 3e. Run aggregation functions like 'sum' over several rows or columns
#i.   sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

#ii.  sum over columns instead:
genre_count = data[genres].sum(axis=1)
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

# 3f. Explore Group Properties
#i.   split up movies by decade
decade = (data.year // 10)*10
tyd = data[['title', 'year']]
tyd['decade'] = decade
tyd.head()

#ii.  mean score for all movies in each decade
decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.plot(decade_mean.index, decade_mean.values, 'o-', color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

#iii. Compute the scater in each year
grouped_scores = data.groupby(decade).score        #defines the list grouped_scores as the values of the column 'score' after data is organized by decade

mean = grouped_scores.mean()                       #defines 'mean' as the mean of the values of grouped_scores
std = grouped_scores.std()                         #defines 'std' as the standard deviation of the values of grouped_scores

plt.plot(decade_mean.index, decade_mean.values, 'o-', color='r', lw=3, label = 'Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values, (decade_mean - std).values, color = 'r', alpha = .2)
plt.scatter(data.year, data.score, alpha = .04, lw = 0, color = 'k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon = False)

#iv.  Iterate over a GroupBy object to find the most popular movie each year:
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values




plt.scatter(data.runtime, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Movie Length")
plt.ylabel("IMDB Rating")
#------Analysis--------
#Most movies, regardless of rating, were about 100 minutes long. The lowest rated movies looked to be just under 100 minutes


plt.scatter(data.year, data.runtime, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("Movie Length")

decadeRuntime = data.groupby(decade).runtime.mean()
print decadeRuntime
#------Analysis--------
#the average movie length hasn't changed appreciably between decades

'''
Trying to find the average score for each genre, gave up b/c it was taking too long to figure out
columnNames = list(data.columns.values)
genreNames = columnNames[7:len(columnNames)]
len(genreNames)
data.groupby(genreNames[1]).head(1)['score']
for n in genreNames:
    data.groupby(n)
'''