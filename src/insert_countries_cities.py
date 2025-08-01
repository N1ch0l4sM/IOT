import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import cities,PG1_CONFIG

import psycopg2

def insert_countries_cities():
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            dbname=PG1_CONFIG['database'],
            user=PG1_CONFIG['user'],
            password=PG1_CONFIG['password'],
            host=PG1_CONFIG['host'],
            port=PG1_CONFIG['port']
        )
        cursor = conn.cursor()

        # Insert countries and cities
        for city in cities:
            country_name = city["country"]
            city_name = city["city"]
            lat = city["lat"]
            lon = city["lon"]

            # Insert country if not exists
            cursor.execute("""
                INSERT INTO country (CountryName)
                VALUES (%s)
                ON CONFLICT (CountryName) DO NOTHING
                RETURNING IdCountry;
            """, (country_name,))
            country_id = cursor.fetchone()
            if country_id is None:
                cursor.execute("SELECT IdCountry FROM country WHERE CountryName = %s;", (country_name,))
                country_id = cursor.fetchone()[0]
            else:
                country_id = country_id[0]

            # Insert city
            cursor.execute("""
                INSERT INTO city (CityName, IdCountry, Lon, Lat)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (CityName, IdCountry) DO NOTHING;
            """, (city_name, country_id, lon, lat))

        # Commit the transaction
        conn.commit()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    insert_countries_cities()