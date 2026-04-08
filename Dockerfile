# Use a stable Python base image (full version for build reliability)
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Removed hardcoded MODEL_NAME to allow for automatic free model discovery

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip first to ensure better dependency resolution
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# HF port
EXPOSE 7860

# Command to run the application
# We use python -m server.app which calls the main() function
CMD ["python", "server/app.py"]
