

#Project milestone 1
##The plan

I plan to forecast bikeshare demand in Washington D.C. (Capitol Bikeshare system). This is a Kaggle playground project titled "Forecast use of a city bikeshare system." 

##The dataset 

I have two datasets containing information about weather and time, each spanning the course of two years. The first dataset contains data for the first 19 days of a given month, and has data on bike demand; it will be the dataset I use to develop my forecasting model. The second dataset contains all the rest of the data for the month, but without any information related to demand demand; it is the dataset that will be sent and scored by Kaggle (using RMSLE) after I estimate fitted demand for it. 


##Data facts and caveats

I know I will have to, at least, process the time variable in the dataset, so to make useful statements about how different aspects of time (hour, day of week, etc.) affect demand.

For now I know that demand is the highest during the summer, and lowest in winter. 

##Why I'm interested
I chose this topic because it gives me the chance to explore a variety of forecasting tools. 