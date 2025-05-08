# Use a base image with Python 3.10
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining app files (code, model, etc.)
COPY . .

# 🚫 Removed heavy data folder to reduce memory usage
# COPY src/data/processed /app/src/data/processed  ← Removed

# Expose the port used by FastAPI
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
