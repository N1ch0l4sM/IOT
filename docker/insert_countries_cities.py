from src.utils.database import db_connection
from src.config import cities

def insert_countries_cities():
    try:
        # Insert countries and cities
        for city in cities:
            country_name = city["country"]
            city_name = city["city"]
            lat = city["lat"]
            lon = city["lon"]

            # Insert country if not exists
            insert_country_query = """
                INSERT INTO country (CountryName)
                VALUES (:country_name)
                ON CONFLICT (CountryName) DO NOTHING
                RETURNING IdCountry;
            """
            params = {"country_name": country_name}
            result = db_connection.execute_query(insert_country_query, db="iot_weather_db", params=params)
            if result.empty:
                select_country_query = "SELECT IdCountry FROM country WHERE CountryName = :country_name;"
                result = db_connection.execute_query(select_country_query, db="iot_weather_db", params=params)
                country_id = result.iloc[0, 0]
            else:
                country_id = result.iloc[0, 0]

            # Insert city
            insert_city_query = """
                INSERT INTO city (CityName, IdCountry, Lon, Lat)
                VALUES (:city_name, :country_id, :lon, :lat)
                ON CONFLICT (CityName, IdCountry) DO NOTHING;
            """
            city_params = {
                "city_name": city_name,
                "country_id": country_id,
                "lon": lon,
                "lat": lat
            }
            db_connection.execute_query(insert_city_query, db="postgres", params=city_params)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    insert_countries_cities()
