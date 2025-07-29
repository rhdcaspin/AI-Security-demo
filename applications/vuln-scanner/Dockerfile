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
RUN mkdir -p /app/scan_results

# Copy application code
COPY scanner_service.py .

# Create non-root user
RUN useradd -m -u 1001 scanner
RUN chown -R scanner:scanner /app

# Switch to non-root user
USER scanner

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Run the application
CMD ["python", "scanner_service.py"] 