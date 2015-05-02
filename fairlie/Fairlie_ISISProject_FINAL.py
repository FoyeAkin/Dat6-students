# -*- coding: utf-8 -*-
"""
Created on Fri May 01 11:24:58 2015

@author: fairlieb
"""
import datetime
import pandas

#Set encoding to UTF-8 if Arabic characters are expected (remove # to activate)
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

#Read in Sysomos file
df = pandas.read_csv('C:/Users/fairlieb/Desktop/Analytics/DAT6/DAT6-master/data/sysomos-content-2015-04-27.csv')

#Define datasets
df['time'] = pandas.to_datetime(df.time,format="%H:%M:%S")
df['counter'] = 1
tweets = df[df['type'] == "TWITTER"]
#df = df[df.text.str.contains("in|the|Iraq|Syria|Islamic State|states",flags = re.IGNORECASE )==True]
df.shape


import dateutil.parser
import pytz

#Format the time for each Tweet
tweets['time'] = tweets.time.apply(lambda x: x.tz_localize('UTC').tz_convert("EST"))
tweets['time'] = tweets.time.apply(lambda fulldate: str(fulldate.time()))
tweets['time'] = pandas.to_datetime(tweets.time,format="%H:%M:%S")
tweets.index = tweets['time']
#Group the Tweets into 30 minute intervals
test = tweets['counter'].resample("30 Min",how='sum')
test=test.sort_index()

#plots the data on a line chart
import matplotlib.pyplot as plt

x = test.index.to_datetime() #map(lambda x: datetime.strftime(x, "%I:%M %p" ), test.index.to_datetime())
y = test

fig, ax = plt.subplots()
ax.plot_date(x, y, linestyle='-')


#Break times into 30 minute intervals
#Add chart labels
ax.set_ylabel("Number of Tweets")
ax.set_xlabel("Time of Day: 30 Min Intervals")
ax.set_title("Best Time To Tweet")

fig.autofmt_xdate()
plt.show()

#Writes data to an Excel file
pandas.DataFrame(test).to_excel("besttimes.xlsx")
