__author__ = "Sahil Nagpal"

# importing the libraries
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, collect_list, udf, mean
from pyspark.sql.types import ArrayType, IntegerType

# creating the spark session
# creating for the local/cluster run as well
spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("PayTM Test Assignment") \
    .getOrCreate()

# step-1
# loading the station data
# using the read method to load the data (gzip files read) , option (header,true) since data as headers
stationDataFrame = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .load("../../paytmteam-de-weather-challenge-beb4fc53605c/stationlist.csv")

# step-1
# loading the countries data
# using the read method to load the data (gzip files read) , option (header,true) since data as headers
countriesDataFrame = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .load("../../paytmteam-de-weather-challenge-beb4fc53605c/countrylist.csv")

# step-1
# loading the weather data
# using the read method to load the data (gzip files read)
weatherDataFrame = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .load("../../paytmteam-de-weather-challenge-beb4fc53605c/data/2019/*.csv.gz")

# ------------------------------------Question-1---------------------------------------------------------
# Which country had the hottest average mean temperature over the year?
# joining station and countries data
countriesDataFrame = countriesDataFrame.withColumnRenamed("COUNTRY_ABBR",
                                                          "CNTRY_ABBR")  # renaming the column to handle the ambiguity issue
joinType = "left"
joinColumn = stationDataFrame.COUNTRY_ABBR == countriesDataFrame.CNTRY_ABBR
countryStationDataFrame = stationDataFrame.join(countriesDataFrame, joinColumn, joinType) \
    .select("STN_NO", "COUNTRY_ABBR", "COUNTRY_FULL")

# joining countryStationDataFrame and weatherData data
weatherDataFrame = weatherDataFrame.withColumnRenamed("STN---", "STN")
joinColumnFinal = weatherDataFrame.STN == countryStationDataFrame.STN_NO
finalDataFrame = weatherDataFrame.join(countryStationDataFrame, joinColumnFinal, joinType) \
    .drop("STN_NO")

finalDataFrame \
    .groupBy(col("COUNTRY_FULL")) \
    .agg(mean("TEMP").alias("AverageMeanTemp")) \
    .alias("AverageMeanTemp") \
    .orderBy(col("AverageMeanTemp").desc()) \
    .show(1)
#
# # +------------+-----------------+
# # |COUNTRY_FULL|  AverageMeanTemp|
# # +------------+-----------------+
# # |    DJIBOUTI|90.06114457831325|
# # +------------+-----------------+

# # ------------------------------------Question-3---------------------------------------------------------
# # Which country had the second highest average mean wind speed over the year?
finalDataFrame \
    .groupBy(col("COUNTRY_FULL")) \
    .agg(mean("WDSP").alias("AverageMeanWindSpeed")) \
    .alias("AverageMeanWindSpeed") \
    .orderBy(col("AverageMeanWindSpeed").desc()) \
    .show(2)

# +------------+--------------------+
# |COUNTRY_FULL|AverageMeanWindSpeed|
# +------------+--------------------+
# |     ARMENIA|   457.3659318266033|
# +------------+--------------------+
# # ------------------------------------Question-2---------------------------------------------------------
# # Which country had the most consecutive days of tornadoes/funnel cloud formations?

# substring FRSHTT for Tornadoes/Funnel
finalDataFrame = finalDataFrame\
    .withColumn("Tornadoes/Funnel",substring('FRSHTT',6,1))

finalDataFrame = finalDataFrame\
    .select("COUNTRY_FULL","Tornadoes/Funnel")\

finalDataFrame = finalDataFrame\
    .withColumn("Tornadoes/Funnel",finalDataFrame['Tornadoes/Funnel'].cast(IntegerType()))

finalDataFrame = finalDataFrame\
    .groupBy(col("COUNTRY_FULL"))\
    .agg(collect_list(col("Tornadoes/Funnel")).alias("Tornadoes/Funnel-Range"))\
# +--------------------+----------------------+
# |        COUNTRY_FULL|Tornadoes/Funnel-Range|
# +--------------------+----------------------+
# |        SOUTH AFRICA|  [0, 0, 0, 0, 0, 0...|
# |             ARMENIA|  [0, 0, 0, 0, 0, 0...|
# |               BURMA|  [0, 0, 0, 0, 0, 0...|
# |            CAMBODIA|  [0, 0, 0, 0, 0, 0...|

#creating and registering the udf to convert the collect list into int
def returnInt(arr):
    return list(map(int, arr))

returnInt = udf(lambda y : returnInt(y),ArrayType(IntegerType()))

#creating and registering the udf to calculate the consecutive
def maxConsecutive(arr):
    count = 0
    result = 0
    for i in range(0, len(arr)):
        if arr[i] == 0:
            count = 0
        else:
            count += 1
            result = max(result, count)

    return result
maxConsecutive = udf(lambda z : maxConsecutive(z))

finalDataFrame\
    .select(col("COUNTRY_FULL"),\
            maxConsecutive(col("Tornadoes/Funnel-Range")).alias("Tornadoes/Funnel-Range"))\
    .show()

