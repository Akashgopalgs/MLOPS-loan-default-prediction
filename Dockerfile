# Use Python 3.10
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install curl (if you ever need it), strip cache
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copy all your source, including the correct model
COPY . .

# Upgrade pip and install dependencies (pin joblib in requirements.txt)
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose and run
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
