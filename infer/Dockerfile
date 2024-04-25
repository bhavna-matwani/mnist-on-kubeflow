FROM python:3.9-slim

# Set up a working folder
WORKDIR /app

# Install necessary system packages using Debian's package manager
RUN apt-get update && apt-get install -y \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the Python script into the container at /app
ADD templates  /app/templates
ADD images  /app/images
ADD main.py  /app/main.py

EXPOSE 8000

# Command to run the application
CMD ["python3", "main.py"]