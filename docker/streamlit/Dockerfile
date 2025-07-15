FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-streamlit.txt /app/requirements-streamlit.txt
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Expose port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
