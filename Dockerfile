# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 8000

# Force logs to show up immediately
ENV PYTHONUNBUFFERED=1

# Run FastAPI
CMD ["uvicorn", "flaskapp:app", "--host", "0.0.0.0", "--port", "8000"]