import requests
import json
from datetime import datetime
import sys
sys.path.append('/home/nicholas/Documents/IOT')
import config as config
from pymongo import MongoClient

# Log the Python executable path and environment variables
print(f"Script started at {datetime.now()}")
# print(f"Python executable: {sys.executable}")
# print(f"Environment variables: {os.environ}")

# OpenWeatherMap API setup
api_key = config.api_key
cities  = config.cities

MONGO_CONFIG = {
    "host": config.mongo_host,
    "port": config.mongo_port,
    "dbname": config.mongo_db,
    "username": config.user,
    "password": config.password,
}

def get_weather(lat, lon, api_key):
    # Call current weather data
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        save_to_mongo(data)
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    
def save_to_mongo(data):
    try:
        client = MongoClient(
            MONGO_CONFIG["host"],
            MONGO_CONFIG["port"],
            username=MONGO_CONFIG["username"],
            password=MONGO_CONFIG["password"],
        )
        db = client[MONGO_CONFIG["dbname"]]
        collection = db.iot_weather
        collection.insert_one(data)
        #print("Data saved to MongoDB")
    except Exception as e:
        print(f"Failed to save data to MongoDB: {e}")

if __name__ == "__main__":
    for city_info in cities:
        lat = city_info["lat"]
        lon = city_info["lon"]
        get_weather(lat, lon, api_key)
    print(f"Script ended at {datetime.now()}")
