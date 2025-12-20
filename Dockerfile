FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application folder for the container
COPY webapp/ webapp/

# Copy the data folder
COPY data/ data/

# Expose the port Flask runs on
EXPOSE 5000

# Run with Gunicorn instead of Flask dev server
# the -w 1 means 1 worker process
# the -b means bind address and btw 127.0.0.1 = container only host machine cannot reach it
# We basically have Browser → Host → Docker → Gunicorn → Flask
CMD [ "gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "webapp.app:app" ]