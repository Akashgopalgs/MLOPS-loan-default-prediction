# Updated Dockerfile
FROM python:3.10-alpine

# Set working directory
WORKDIR /app

# Install build dependencies (required for some pip packages)
RUN apk add --no-cache build-base

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY app.py ./
COPY static ./static
COPY templates ./templates
COPY src/models ./src/models

# Expose port used by FastAPI
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
