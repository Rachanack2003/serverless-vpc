# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (including Tesseract OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project, including gemini-sa-key.json
COPY . .


# Expose the port for Flask
EXPOSE 8080

# Command to run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
