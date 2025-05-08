# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional but useful for some environments)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Upgrade pip and install specific versions of scikit-learn and joblib
RUN pip install --upgrade pip \
 && pip install scikit-learn==1.6.1 joblib==1.4.2

# Install remaining Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
