FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for API
EXPOSE 8000

# Default command runs training then serves API
# Can be overridden at runtime
CMD ["sh", "-c", "python src/models/train.py && uvicorn src.backend.api:app --host 0.0.0.0 --port 8000"]
