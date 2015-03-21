Lena Nguyen | GA DAT-6 | 6 March 2015

# Course Project

## Question
For my final project, I'm planning use data build a model that will forecast Metro rail ridership.

## Motivation
I'm interested in topics relating to transportation, urban planning, and more generally, about how people move around a city and what form(s) of transportation they use.

## Data
### Historic Metrorail ridership data
I plan to use historic Metrorail ridership data to validate the model. 

WMATA provides [historic ridership data](http://www.wmata.com/pdfs/planning/Historical%20Rail%20Ridership%20By%20Station.pdf) on their website by station and year from 1977 to 2011. It is unfortunately provided in PDF format.

But luckily, Open Data DC also has [historic ridership data](http://www.opendatadc.org/dataset/wmata-metrorail-ridership-by-date) for every day in the last 10 years. It is in a much more palatable csv form. The data is simple and there are no missing values. It only has two columns: One for the date and the other is number of riders. It appears that the number of riders have been rounded off to the near hundred thousand because it was pulled from a data visualization off the Plan It Metro website. I also see outliers due to major events (ie Presidential inaugrations). I think I will have to aggregate this data to the month level to get all the data at the same level. 

### Data for factors affecting ridership
These are factors I've thought of so far. Will add more if I think of more.

* Fare prices: Available on the WMATA website

* Gas prices: The US Energy Information Administration has [historic gas prices](http://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPMRR_PTE_R1Z_DPG&f=W) available in xls format for the every week from 1995 to present day for regions in the US. DC/MD/VA is part of the Lower Atlantic region. Data is very simple and clean. It just has the week in one column and the price of gas in USD per gallon in another. 

* Weather: NOAA has historic [weather data](http://www.ncdc.noaa.gov/cdo-web/datatools) on their website. You can get monthly, daily, and even hourly data from all their weather stations. The weather in the DC metro area does not vary dramatically so I plan to use that data to get the pattern of weather variations. I sent in a request for the csv data from the DC Arboretum station but have not received it yet. From looking at the data online, I can see that it has the data that I need: monthly mean temperature, total precipation, and total snowfall. 

* DC employment figures: The US Bureau of Labor Statistics provide historical [labor force data](http://www.bls.gov/eag/eag.dc_washington_md.htm) for the DC metro area

* Demographics: US Census Bureau has [population data](http://www.census.gov/popest/data/national/totals/2014/index.html). They are broken down by state and are estimated using census data (except for the years when there was a census).

* Tourist numbers?: Smithsonian has [data](http://dashboard.si.edu/at-the-smithsonian) on number of visitors but only by year. DC's tourist website has data on their [PDF presentations](http://washington.org/press/DC-information/washington-dc-visitor-research) but not in any other form. I have not found a good dataset to estimate this yet. 

* Reliability of service?: Open data DC has data about Metro [service disruptions](http://www.opendatadc.org/dataset/wmata-disruption-reports) from April 2012 to present. There is no documentation as to how the data was compiled. The data has a lot of text describing what the incident was. I don't really care about what type of incident or what the cause was but that there was one and it caused a delay in service. There is also a delay column that I'm assuming is an estimate on the number of minutes it delayed the trains by. Unfortunately, this data only goes back to 2012 so I might not be able to use it as part of my model.




