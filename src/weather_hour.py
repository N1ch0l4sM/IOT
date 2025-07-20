import sys
sys.path.append('/home/nicholas/projects/IOT')
import config as config

from pymongo import MongoClient
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MongoDB to PostGres") \
    .config("spark.mongodb.input.uri", f"mongodb://{config.user}:{config.password}@{config.mongo_host}:{config.mongo_port}/{config.mongo_db}.iot_weather?authSource=admin") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.postgresql:postgresql:42.6.0") \
    .getOrCreate()

# MongoDB connection setup
mongo_host = config.mongo_host
mongo_port = config.mongo_port
mongo_db = config.mongo_db
mongo_collection = config.mongo_collection

# PostgreSQL connection properties
postgres_properties = {
    "user": config.user,
    "password": config.password,
    "driver": "org.postgresql.Driver"
}

# Connect to MongoDB
client = MongoClient(mongo_host, mongo_port, username=config.user, password=config.password, authSource="admin")
db = client[mongo_db]
collection = db[mongo_collection]

# Calculate the timestamp for one hour ago
one_hour_ago_timestamp = (datetime.now() - timedelta(hours=1)).timestamp()

# Load all data from MongoDB without filtering using DataFrame API
weather_data = spark.read.format("mongo") \
    .option("database", config.mongo_db) \
    .option("collection", config.mongo_collection) \
    .load()

weather_data.createOrReplaceTempView("weather_raw")

# Use SparkSQL to filter data from one hour ago and compute aggregates
avg_temps_sql = f"""
WITH filtered_weather AS (
    SELECT * FROM weather_raw WHERE dt >= {one_hour_ago_timestamp}
)
SELECT 
    ROUND(coord.lon, 2) AS lon, 
    ROUND(coord.lat, 2) AS lat,
    ROUND(AVG(main.temp), 2) AS avg_temperature,
    ROUND(AVG(main.humidity), 2) AS avg_humidity,
    ROUND(AVG(COALESCE(rain.`1h`, 0)), 2) AS avg_rain,
    ROUND(AVG(wind.speed), 2) AS avg_wind,
    ROUND(AVG(clouds.all), 2) AS avg_clouds,
    ROUND(AVG(main.pressure), 2) AS avg_pressure,
    ROUND(AVG(main.feels_like), 2) AS avg_feels_like
FROM filtered_weather
GROUP BY ROUND(coord.lon, 2), ROUND(coord.lat, 2)
"""
avg_temps = spark.sql(avg_temps_sql)
avg_temps.createOrReplaceTempView("avg_weather")

# Read city data from PostgreSQL using Spark JDBC
city_df = spark.read \
    .jdbc(
        url=f"jdbc:postgresql://{config.postG_host}:{config.postG_port}/{config.postG_db}",
        table="city",
        properties=postgres_properties
    )
city_df.createOrReplaceTempView("cities")

# SparkSQL join query to match city data with weather averages
joined_sql = """
SELECT 
    c.idCity AS city_id,
    c.CityName AS city_name,
    w.avg_temperature,
    w.avg_humidity,
    w.avg_rain,
    w.avg_wind,
    w.avg_clouds,
    w.avg_feels_like,
    w.avg_pressure,
    w.lon,
    w.lat
FROM avg_weather w
JOIN cities c 
    ON ROUND(w.lat, 2) = ROUND(c.lat, 2)
    AND ROUND(w.lon, 2) = ROUND(c.lon, 2)
"""
result_df = spark.sql(joined_sql)
result_df.createOrReplaceTempView("result_join")

# Compute one-hour-ago date and hour in Python
one_hour_ago = datetime.now() - timedelta(hours=1)
current_date = one_hour_ago.strftime("%Y-%m-%d")
one_hour_num = one_hour_ago.hour

# Use SparkSQL to add Date and Hour columns and select final mapping
final_sql = f"""
SELECT 
    city_id AS idCity,
    CAST('{current_date}' AS date) AS Date,
    {one_hour_num} AS Hour,
    avg_temperature AS Temp,
    avg_feels_like AS FeelsLike,
    avg_clouds AS Clouds,
    avg_rain AS Rain,
    avg_wind AS Wind,
    avg_pressure AS Pressure,
    avg_humidity AS Humidity
FROM result_join
"""
final_df = spark.sql(final_sql)

# PostgreSQL connection properties and URL
jdbc_url = f"jdbc:postgresql://{config.postG_host}:{config.postG_port}/{config.postG_db}"
postgres_properties = {
    "user": config.user,
    "password": config.password,
    "driver": "org.postgresql.Driver"
}

try:
    # Show the joined results
    final_df.show()
    
    # Write the final result to the weather_hour table using Spark SQL result
    final_df.write.mode("append").jdbc(url=jdbc_url, table="weather_hour", properties=postgres_properties)
    
    print("Successfully appended records to weather_hour table using PySpark")
except Exception as e:
    print("Error occurred:", e)
finally:
    # Close the MongoDB connection and stop the Spark session
    client.close()
    spark.stop()