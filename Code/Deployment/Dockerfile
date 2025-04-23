# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED 1
# Set the listening port
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir helps keep image size down slightly
# Use the specified index URL for torch CPU versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# Includes app.py, model files, class names, templates directory
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define the command to run the application using Gunicorn
# Using 2 workers as a starting point, adjust based on Cloud Run instance size/load
# Ensure app:app matches your flask file name (app.py) and Flask app object name (app)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]