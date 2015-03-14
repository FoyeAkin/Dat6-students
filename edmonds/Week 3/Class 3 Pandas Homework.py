import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Tell pandas to display wide tables as pretty HTML tables
#I know the instructions said not to, but I just wanted my tables to look like theirs
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

#Setting a variable to use as the first argument in read_csv()
data_URL = 'http://bit.ly/cs109_imdb'

names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv(data_URL, delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows

clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

#Determine the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

#Make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

#Removing the redundant years from movie titles
data['title'] = [t[0:-7] for t in data.title]
data.head()

data[['score', 'runtime', 'year', 'votes']].describe()

#Hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

#Flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

data.runtime.describe()
#Minimum is now 45 minutes

plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")
remove_border()

plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")
remove_border()

plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")
remove_border()

plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")
remove_border()

plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')
remove_border()

# Low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})
# Not sure why I'm geting numbers instead of genre names...

genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

decade =  (data.year // 10) * 10

tyd = data[['title', 'year']]
tyd['decade'] = decade

tyd.head()

#Mean score for all movies in each decade
decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
remove_border()

grouped_scores = data.groupby(decade).score

mean = grouped_scores.mean()
std = grouped_scores.std()

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
remove_border()

for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values
    
#---------------------Homework---------------------#

#Since their code for providing genre counts didn't work for me,
#I'll try it a different way
genre_counts = pd.DataFrame(index=np.array(genres), columns=np.array(['Count']))
for genre in genres:
    genre_counts.set_value(genre, 'Count', (data[genre]==True).sum())
genre_counts.sort_index(by='Count', ascending=False, inplace=True)
genre_counts

#Checking worst movies by year, just out of curiosity
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.min()].title.values

#I'd like to see how much the worst movies change with a minimum vote threshold
for year, subset in data[data.votes >= 5000].groupby('year'):
    print year, subset[subset.score == subset.score.min()].title.values

for year, subset in data[data.votes >= 10000].groupby('year'):
    print year, subset[subset.score == subset.score.min()].title.values
    
#Creating DataFrames as an easy way to view worst films with their scores and vote counts
#Though maybe not the most elegant, the following should work even if there are ties
worst = pd.DataFrame(index=np.arange(1950,2012), columns=np.array(['Film', 
                     'Score', 'Votes']))
for year, subset in data.groupby('year'):
    worst.set_value(year, 'Film', subset[subset.score == subset.score.min()].title.values)
    worst.set_value(year, 'Score', subset.score.min())
    worst.set_value(year, 'Votes', subset[subset.score == subset.score.min()].votes.values)
    
#Minimum of 5000 votes
worst_5k = pd.DataFrame(index=np.arange(1950,2012), columns=np.array(['Film', 
                     'Score', 'Votes']))
for year, subset in data[data.votes >= 5000].groupby('year'):
    worst_5k.set_value(year, 'Film', subset[subset.score == subset.score.min()].title.values)
    worst_5k.set_value(year, 'Score', subset.score.min())
    worst_5k.set_value(year, 'Votes', subset[subset.score == subset.score.min()].votes.values)

#Minimum of 10000 votes
worst_10k = pd.DataFrame(index=np.arange(1950,2012), columns=np.array(['Film', 
                     'Score', 'Votes']))
for year, subset in data[data.votes >= 10000].groupby('year'):
    worst_10k.set_value(year, 'Film', subset[subset.score == subset.score.min()].title.values)
    worst_10k.set_value(year, 'Score', subset.score.min())
    worst_10k.set_value(year, 'Votes', subset[subset.score == subset.score.min()].votes.values)

#-------------------Homework Plots-------------------#

'''
Of the above DataFrames, worst_10k is the only one with unique results for each year.
Since the numbers of votes are currently stored as arrays, I'm going to convert them
to integers before trying to plot.
'''
for year in worst_10k.index:
    worst_10k.Votes[year] = worst_10k.Votes[year][0]
worst_10k.plot(kind='Scatter', x='Score', y='Votes', alpha=0.3,
               title="Scores of Each Year's Worst Film by Number of Votes (min. 10,000)")
plt.savefig('scores_vs_votes_worst.png')

#Are people more likely to rate a movie when they like it or dislike it?
data.groupby('score').votes.mean().plot(kind='bar', color='b', 
                                        title='Score by Average Number of Votes')
plt.savefig('scores_by_votes.png')
#Much more likely to vote when they like a movie!
#Because the file isn't too clear, I'm also adding a screenshot of an expanded version.