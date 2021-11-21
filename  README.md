# Zephrus
Nov 2021. It's time to at least partly publish what I have been upto. 
This is the base for all things research I have been upto in the last little bit.


## Data 
The data for the project exists in three major forms:

- Irradiance data for 23 plants 
- Commercial weather data (point based)
- Satellite weahter data / imagery


### Irradiance data
Currently, we are unable to publish the irradiance data however we are working on it.


### Point based weather data

We have used [weatherbit](https://www.weatherbit.io/), a commercial weather data supplier, to access historic weather data covering the area of our plants.
Data can be accessed using their APIs.

### Satellite data
We have used a few sources of satellite imagery

#### MetOffice
The UK met office provided satellite and radar imagery covering the UK. 
This is provided in near real time and can be accessed using their APIs.
Unfortunately, when last checked there was no historical data.

#### EUMetSat
EUMetSat are one of the sources of satellite weather data that we plan to use.
Their data is a new addition to the project that we hope will be bennificail.
The folder `eumetsat` has all the code needed to load the eumetsat data into a ADLSv1 data lake.
This will be updated to Gen2 soon as we plan to migrate all the cloud infra to newer Azure services.
To get all imagery to the UK it is about ~6TB raw, once processed it is only ~60GB

#### Open Weather Map
[Open Weather Map](), Another commercial supplier they offer weather imagery for the whole globe.
This data can be accessed for a few years of historical data. 