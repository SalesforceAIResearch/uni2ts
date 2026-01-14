FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server script
COPY server.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SANDBOX_PORT=8080

# Expose the port
EXPOSE 8080

# Run the HTTP server
CMD ["python", "server.py"]

