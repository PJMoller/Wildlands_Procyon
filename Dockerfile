FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application folder for the container
COPY webapp/ webapp/

# Copy the src folder (needed for imports!)
COPY src/ src/

# Copy the data folder
COPY data/ data/

# Set PYTHONPATH so Python can find the src module
ENV PYTHONPATH=/app

# Expose the port Flask runs on
EXPOSE 5000

# Run with Gunicorn instead of Flask dev server
# Increased timeout to 120s for slow ML predictions
# Increased workers to 2 for better performance
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "webapp.app:app"]
