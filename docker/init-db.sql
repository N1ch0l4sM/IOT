-- Initialize Weather Database
-- The database weather_db is created via environment variable in docker-compose.yml

-- Create tables for weather data
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    location VARCHAR(100) NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    wind_direction VARCHAR(10),
    precipitation FLOAT,
    rain_probability FLOAT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    weather_data_id INTEGER REFERENCES weather_data(id),
    prediction BOOLEAN NOT NULL,
    confidence FLOAT,
    model_version VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_weather_data_location ON weather_data(location);
CREATE INDEX IF NOT EXISTS idx_weather_data_recorded_at ON weather_data(recorded_at);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at);

-- Insert sample data
INSERT INTO weather_data (location, temperature, humidity, pressure, wind_speed, wind_direction, precipitation, rain_probability)
VALUES 
    ('São Paulo', 25.5, 65.0, 1013.2, 12.5, 'NE', 0.0, 0.15),
    ('Rio de Janeiro', 28.0, 70.0, 1010.5, 8.2, 'E', 0.0, 0.25),
    ('Belo Horizonte', 22.8, 60.0, 1015.8, 10.1, 'SW', 0.0, 0.10);
