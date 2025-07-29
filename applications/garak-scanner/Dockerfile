FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application directories
RUN mkdir -p /app/garak_configs /app/scan_results

# Copy application code
COPY garak_service.py .

# Create non-root user
RUN useradd -m -u 1001 garak
RUN chown -R garak:garak /app

# Switch to non-root user
USER garak

# Expose port
EXPOSE 5004

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5004/health || exit 1

# Run the application
CMD ["python", "garak_service.py"] 