'''
-----------------------------------------------------------------------------
--Positions in Today's NBA: Are Traditional Classifications Still Relevant?--
---------------------DAT6 Student Project (First Draft)----------------------
--------------------------------Joe Edmonds----------------------------------
------------------------------April 11, 2015---------------------------------
_____________________________________________________________________________
'''

#-----------------Step 1: Importing relevant Libraries-----------------#
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics



#-----------------Step 2: Data Ingestion and Cleaning------------------#
# Data obtained from www.basketball-reference.com

# 2006-14: http://www.basketball-reference.com/leagues/NBA_2007_advanced.html
#          through http://www.basketball-reference.com/leagues/NBA_2014_advanced.html
old = pd.read_csv('/Users/josephedmonds/Desktop/nba0607.csv')
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba0708.csv')
old = old.append(old1, ignore_index=True)
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba0809.csv')
old = old.append(old1, ignore_index=True)
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba0910.csv')
old = old.append(old1, ignore_index=True)
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba1011.csv')
old = old.append(old1, ignore_index=True)
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba1112.csv')
old = old.append(old1, ignore_index=True)
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba1213.csv')
old = old.append(old1, ignore_index=True)
old1 = pd.read_csv('/Users/josephedmonds/Desktop/nba1314.csv')
old = old.append(old1, ignore_index=True)
del old1
# Is there a cleaner way to get this same DataFrame?

# 2014-15: http://www.basketball-reference.com/leagues/NBA_2015_advanced.html
new = pd.read_csv('/Users/josephedmonds/Desktop/nba1415.csv')

# Drop irrelevant columns (These include rank, age, team, games played, 
# efficiency rating, win shares, +/- ratings, and other advanced metrics)
to_drop = [0,3,4,5,7,13,19,20,21,22,23,24,25,26,27,28]
old.drop(old.columns[to_drop], axis=1, inplace=True)
new.drop(new.columns[to_drop], axis=1, inplace=True)
del to_drop

# Rename remaining columns
new_cols = ['Name','Position','MP','TS','3PR','FTR','ORB','DRB','AST',
            'STL','BLK','TO','USG']
'''
Definitions of Chosen Metrics:
MP  = Minutes played (over course of season)

TS  = "True shooting percentage" (PTS / 2(FGA + 0.44*FTA), where PTS = 
       points scored, FGA = field goal attempts, and FTA = free throw attempts)
       
3PR = Three-pointer rate (percentage of FGAs from three-point range)

FTR = Free throw rate (# of FTAs per FGA)

ORB = Offensive rebounding rate (percentage of available offensive rebounds
      player grabbed while on floor)
      
DRB = Defensive rebounding rate (percentage of available defensive rebounds
      player grabbed while on floor)
      
AST = Assist rate (percentage of teammate FGs player assisted while on floor)

STL = Steal rate (percentage of opponent possessions that end with a steal
      while player is on floor)

BLK = Block rate (percentage of opponent FGAs player blocked while on floor)

TO  = Turnover rate (TOs committed per 100 plays)

USG = Usage rate
'''
old.columns = new_cols
new.columns = new_cols
del new_cols

# Drop players who are listed under more than one position
Pos_len = [len(old.Position[x]) for x in range(len(old.Position))]
old['PL'] = Pos_len
old = old[old.PL <= 2]
old.drop(['PL'], axis=1, inplace=True)

Pos_len = [len(new.Position[x]) for x in range(len(new.Position))]
new['PL'] = Pos_len
new = new[new.PL <= 2]
new.drop(['PL'], axis=1, inplace=True)
del Pos_len, x

# Change position markers to numeric equivalents
positions = {'PG':1, 'SG':2, 'SF':3, 'PF':4, 'C':5}
old['Position'] = old['Position'].map(positions)
new['Position'] = new['Position'].map(positions)
del positions

# Limit this year's data to top 300 qualifying players in minutes played,
# then delete that column, which is no longer needed.
new.index = range(len(new.index))
new = new[new.index < 300]
new.drop(['MP'], axis=1, inplace=True)

# Limit older data to players who logged more than 800 minutes on the season,
# roughly the same range as this year's data, then delete that column.
old = old[old.MP > 800]
old.drop(['MP'], axis=1, inplace=True)


# Convert all columns with percentage values to decimals
'''
NOTE: I'm commenting out this block of code, because converting everything
      to decimals gave worse results, for reasons that aren't clear to me...
percent = ['ORB','DRB','AST','STL','BLK','TO','USG']
for col in percent:
    decimal  = [old[col][x]/100 for x in range(300)]
    decimal2 = [new[col][x]/100 for x in range(300)]
    old[col] = decimal
    new[col] = decimal2
del col, percent, decimal, decimal2, x
'''

# Split the old data into train and test sets
features = ['TS','3PR','FTR','ORB','DRB','AST','STL','BLK','TO','USG']
X = old[features]
y = old.Position
X_train, X_test, y_train, y_test = train_test_split(X, y)
train = pd.DataFrame(data=X_train, columns=features)
train['Position'] = y_train
test  = pd.DataFrame(data=X_test, columns=features)
test['Position']  = y_test
# NOTE: I haven't had reason to use this yet, because the point of what I've 
# done so far was not to find the most accurate model, trusting instead that I 
# would be "teaching" the Naive Bayes classifier with enough data that the
# information it would give me on this year's players' positions would be 
# accurate. The above train/test sets are for future exporation.



#---------------------Step 3: Building a Model-------------------------#

nb = MultinomialNB()
nb.fit(X, y)
preds = nb.predict(new[features])

print metrics.accuracy_score(new.Position, preds)
# 0.64 seems low, but that's actually to be expected!
# Few NBA players are a "pure" version of their listed position, and 
# our purpose is to find out what these results indicate about whether the
# way we classify positions aligns with the way the game is now played.

# Making a DataFrame that lists each player and the model's predicted 
# probability that he plays a certain position
probs = nb.predict_proba(new[features])
position_list = ['PG', 'SG', 'SF', 'PF', 'C']
pos = pd.DataFrame(data=probs, columns=position_list)

# Converting probabilities from decimal to percentages
for col in position_list:
    percent = [pos[col][x]*100 for x in range(300)]
    pos[col] = percent

# Including player names and their listed positions
pos['Player'] = new['Name']
pos['Listed as'] = new['Position']

# Converting positional numbers to positional titles
positions = {1:'PG', 2:'SG', 3:'SF', 4:'PF', 5:'C'}
pos['Listed as'] = pos['Listed as'].map(positions)
del positions

# Identifying position with highest probability for each player
predictions = []
for x in range(300):
    pred = np.array(pos[pos.index==x][position_list])
    if np.argmax(pred) == 0:
        predictions.append('PG')
    elif np.argmax(pred) == 1:
        predictions.append('SG')
    elif np.argmax(pred) == 2:
        predictions.append('SF')
    elif np.argmax(pred) == 3:
        predictions.append('PF')
    else:
        predictions.append('C')
pos['Predicted'] = predictions      

# For display purposes, limiting probabilities to four decimal points
for col in position_list:
    percent = ["%.4f" % (pos[col][x]) for x in range(300)]
    pos[col] = percent

# Reordering columns
pos_cols = ['Player', 'Listed as', 'Predicted', 'PG', 'SG', 'SF', 'PF', 'C']
pos = pos[pos_cols]
del pos_cols  

print pos

'''
I have the table of results I was trying to get, but I still have to perform
a good deal of analysis and use the results to expand the scope of the questions
I'm trying to answer, if only because getting this far made me realize that
the results I was trying to get aren't that interesting/useful on their own.

Since there are a number of results that don't make sense to me still (one 
example: Marc Gasol is a C and most definitely plays like a C, so I
can't explain why the classifier would list him as a SF), I'd like to take
another look at the metrics I chose and see if I can improve upon them. I'd
also like to use K-means clustering as an alternative method of classifying
my data, though I'd still have to figure out how best to implement a model like
that with the data I've obtained.

'''
