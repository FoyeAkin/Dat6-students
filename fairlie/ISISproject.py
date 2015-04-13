# -*- coding: utf-8 -*-
"""
Created on Mon Apr 6 00:13:49 2015

@author: fairlieb
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json


#Enter your consumer key, consumer secret, access token, access secret.
consumer_key = "***************************"
consumer_secret = "***************************"
access_token = "***************************"
access_secret = "***************************"


# Search terms, users, or hashtags
#Use http://www.idfromuser.com/ to get user ID
users=["3121049416","2796827393","2772495039","2796827393","2796827393"]
hash_tags=["الدولة_الإسلامية","الدولة_الإسلامية_في_العراق_والشام","#ISIS"]
terms=["الدولة الإسلامية","الموصل‎","الصفويين"]


class Listener(StreamListener):
 def on_data(self, data):
 try:
     
 # Parsing the json received from the twitter stream
 jsonData = json.loads(data)
 print(jsonData)
 createdAt = jsonData['created_at']
 text = jsonData['text']
 print "Created at : " , createdAt , " text : " , text
 saveThis = createdAt + " ---> " + text
 saveFile = open("results.csv","a
 saveFile.write(saveThis.encode('utf-16
 saveFile.write("\n
 saveFile.close()
 return True
 except BaseException , e:
 print 'Failed on data : ', str(e)
 time.sleep(5)

 def on_error(self, status):
 print "Error : ",status

if __name__ == '__main__':
 auth = OAuthHandler(consumer_key, consumer_secret)
 auth.set_access_token(access_token, access_secret)

 # Listener for twitter streaming
 twitterStream = Stream(auth, Listener())

 # Twitter filter
 twitterStream.filter(follow=users,track=hash_tags+ terms,languages=["en"]) 
 
 
 #CALCULATING THE BEST TIME TO TWEET --> Still working on this part!!
tweets['created_moment'] = tweets.created_time.apply(lambda x: x.tz_localize('UTC').tz_convert("EST"))
tweets['created_moment'] = tweets.created_moment.apply(lambda fulldate: str(fulldate.time()))
tweets['created_moment'] = pandas.to_datetime(tweets.created_moment,format="%H:%M:%S")
tweets.index = tweets['created_moment']
test = tweets['counter'].resample("30 Min",how='sum')
test=test.sort_index()

#plots the data on a line chart
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

x = test.index.to_datetime() #map(lambda x: datetime.strftime(x, "%I:%M %p" ), test.index.to_datetime())
y = test

fig, ax = plt.subplots()
ax.plot_date(x, y, linestyle='-')

#ax.annotate('Test', (mdates.date2num(x[1]), y[1]), xytext=(15, 15), 
#            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#Break times into 30 minute intervals
ax.set_ylabel("Number of Tweets")
ax.set_xlabel("Time of Day: 30 Min Intervals")
ax.set_title("Best Time To Tweet")

fig.autofmt_xdate()
plt.show()

#Writes data to an Excel file
pandas.DataFrame(test).to_excel("besttimes.xlsx")