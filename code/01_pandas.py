'''
Data Analysis in Python

    ----UFO data----
    Scraped from: http://www.nuforc.org/webreports.html
    Article Analyzing the Dataset: http://josiahjdavis.com/    
'''

'''
Spyder Reference (Windows):
    Ctrl + Enter    (in the editor):        Runs the line of code
    Ctrl + L        (in the console):       Clears the console
    Up/Down-arrows  (in the console):       Retreives previous commands
    Ctrl + S        (in the editor):        Saves the file              
'''

'''
Reading, Summarizing data
'''

import pandas as pd  # This line imports  (already installed) python package

# Running this next line of code assumes that your console working directory is set up correctly 
# To set up your working directory
#        1) Put the data and the script in the same working directory
#        2) Select the options buttom in the upper right hand cornder of the editor
#        3) Select "Set console working directory"

ufo = pd.read_csv('../data/ufo.csv')   # .. goes up one level in the working directory

ufo                 
ufo.head(15)          # Look at the top x observations
ufo.tail()            # Bottom x observations (defaults to 5)
ufo.describe()        # describe any numeric columns (unless all columns are non-numeric)
ufo.index             # "the index" (aka "the labels")
ufo.columns           # column names (which is "an index")
ufo.dtypes            # data types of each column
ufo.values            # underlying numpy array
ufo.info()            # concise summary

# DataFrame vs Series, selecting a column
type(ufo)
isinstance(ufo, pd.DataFrame)
ufo['State']
ufo.State            # equivalent
type(ufo.State)

# summarizing a non-numeric column
ufo.State.describe()        # Only works for non-numeric if you don't have any numeric data 
ufo.State.value_counts()    # Valuable if you have numeric columns, which you often will

ufo.State.value_counts() / ufo.shape[0] # Values divided by number of records

'''
Slicing / Filtering / Sorting
'''
# selecting multiple columns
ufo[['State', 'City','Shape Reported']]
my_cols = ['State', 'City']
ufo[my_cols]
type(ufo[my_cols])

# loc: filter rows by LABEL, and select columns by LABEL
ufo.loc[1,:]                            # row with label 1
ufo.loc[:3,:]                           # rows with labels 1 through 3
ufo.loc[1:3, 'City':'Shape Reported']   # rows 1-3, columns 'City' through 'Shape Reported'
ufo.loc[:, 'City':'Shape Reported']     # all rows, columns 'City' through 'Shape Reported'
ufo.loc[[1,3], ['City','Shape Reported']]  # rows 1 and 3, columns 'City' and 'Shape Reported'
ufo.loc[[1,3], ['City','Shape Reported']]  # rows 1 and 3, columns 'City' and 'Shape Reported'

# iloc: filter rows by POSITION, and select columns by POSITION
ufo.iloc[0,:]                       # row with 0th position (first row)
ufo.iloc[0:3,:]                     # rows with positions 0 through 2 (not 3)
ufo.iloc[0:3, 0:3]                  # rows and columns with positions 0 through 2
ufo.iloc[:, 0:3]                    # all rows, columns with positions 0 through 2
ufo.iloc[[0,2], [0,1]]              # 1st and 3rd row, 1st and 2nd column

# mixing: select columns by LABEL, then filter rows by POSITION
ufo.City[0:3]
ufo[['City', 'Shape Reported']][0:3]
    
# logical filtering
ufo[ufo.State == 'TX']
ufo[~(ufo.State == 'TX')]   # ufo[(ufo.State == 'TX') == False]
ufo.City[ufo.State == 'TX']
ufo[ufo.State == 'TX'].City             # Same thing
ufo[(ufo.State == 'CA') | (ufo.State =='TX')]
ufo_dallas = ufo[(ufo.City == 'Dallas') & (ufo.State =='TX')]
ufo[ufo.City.isin(['Austin','Dallas', 'Houston'])]

# sorting
ufo.State.order()                               # only works for a Series
ufo.sort_index(inplace=True)                    # sort rows by label
ufo.sort_index(ascending=False)
ufo.sort_index(by='State')                      # sort rows by specific column
ufo.sort_index(by=['State', 'Shape Reported'])  # sort by multiple columns
ufo.sort_index(by=['State', 'Shape Reported'], ascending=[False, True], inplace=True)  # specify sort order

# detecting duplicate rows
ufo.duplicated()                                # Series of logicals
ufo.duplicated().sum()                          # count of duplicates
ufo[ufo.duplicated(['State','Time'])]           # only show duplicates
ufo[ufo.duplicated()==False]                    # only show unique rows
ufo_unique = ufo[~ufo.duplicated()]             # only show unique rows
ufo.duplicated(['State','Time']).sum()          # columns for identifying duplicates

'''
Modifying Columns
'''

# add a new column as a function of existing columns
ufo['Location'] = ufo['City'] + ', ' + ufo['State']
ufo.head()

# rename columns inplace
ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=True)
ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=False)

ufo.head()

# hide a column (temporarily)
ufo.drop(['Location'], axis=1)
ufo[ufo.columns[:-1]]

# delete a column (permanently)
del ufo['Location']

'''
Handling Missing Values
'''

# missing values are often just excluded
ufo.describe()                          # excludes missing values
ufo.Shape.value_counts()                # excludes missing values
ufo.Shape.value_counts(dropna=False)    # includes missing values (new in pandas 0.14.1)

# find missing values in a Series
ufo.Shape.isnull()       # True if NaN, False otherwise
ufo.Shape.notnull()      # False if NaN, True otherwise
ufo.Shape.isnull().sum() # count the missing values
ufo.isnull().sum()
ufo[(ufo.Shape.notnull()) & (ufo.Colors.notnull())]

ufo_shape_not_null = ufo[ufo.Shape.notnull()]

# find missing values in a DataFrame
ufo.isnull()
ufo.isnull().sum()

# drop missing values
ufo.dropna()             # drop a row if ANY values are missing
ufo.dropna(how='all')    # drop a row only if ALL values are missing

# fill in missing values
ufo.Colors.fillna(value='Unknown', inplace=True)
ufo.fillna('Unknown')