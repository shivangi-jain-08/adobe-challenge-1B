FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY extract_outline.py .
COPY document_analyzer.py .
COPY main.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Make the main script executable
RUN chmod +x main.py

# Set the default command
CMD ["python", "main.py"]