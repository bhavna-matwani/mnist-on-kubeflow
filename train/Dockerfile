# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set up a working folder
WORKDIR /app

# Install necessary system packages using Debian's package manager
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container at /app
COPY main.py /app/

# Command to run the application
CMD ["python", "main.py"]