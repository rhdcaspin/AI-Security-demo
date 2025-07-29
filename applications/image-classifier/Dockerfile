FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application directories
RUN mkdir -p /app/uploads /app/models /app/data

# Copy application code
COPY app.py .

# Create non-root user (but still run as root for demo vulnerabilities)
RUN useradd -m -u 1000 appuser
# Note: We intentionally run as root to demonstrate security issues

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"] 