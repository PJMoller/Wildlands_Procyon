FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application folder for the container
COPY webapp/ .

# Copy the data folder
COPY data/ data/

# Expose the port Flask runs on
EXPOSE 5000

# This starts the flask server
CMD [ "python", "app.py" ]