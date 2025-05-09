# Use lightweight Python image
FROM python:3.10-alpine

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary app code and model files
COPY app.py .
COPY src/models /app/src/models

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
