-- Initialize Weather Database
-- The database weather_db is created via environment variable in docker-compose.yml

-- Create Country Table --ok
CREATE TABLE country (
    IdCountry SERIAL PRIMARY KEY,
    CountryName VARCHAR(100) NOT NULL
);

-- Create City Table --ok
CREATE TABLE city (
    IdCity SERIAL PRIMARY KEY,
    CityName VARCHAR(100) NOT NULL,
    IdCountry INTEGER REFERENCES Country(IdCountry) ON DELETE CASCADE,
    Lon FLOAT NOT NULL,
    Lat FLOAT NOT NULL
);

-- Create WeatherHour Table --ok
CREATE TABLE weather_hour (
    IdCity INTEGER REFERENCES City(IdCity) ON DELETE CASCADE,
    Date DATE NOT NULL,
    Hour INT NOT NULL,
    Temp FLOAT,
    FeelsLike FLOAT,
    Clouds FLOAT,
    Rain FLOAT,
    Wind FLOAT,
    Pressure FLOAT,
    Humidity FLOAT
    PRIMARY KEY (IdCity, Date, Hour)
);

-- Add unique constraint to CountryName
ALTER TABLE country
ADD CONSTRAINT unique_country_name UNIQUE (CountryName);

-- Add unique constraint to CityName and IdCountry
ALTER TABLE city
ADD CONSTRAINT unique_city_country UNIQUE (CityName, IdCountry);

-- Indexes for faster queries
CREATE INDEX idx_weather_hour ON weather_hour (IdCity, Date, Hour);


