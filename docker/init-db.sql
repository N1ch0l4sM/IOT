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

