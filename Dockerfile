# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install curl (for downloading the model)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Create the models folder
RUN mkdir -p src/models

# Download the exact model you tested locally 
# directly into src/models so your app loads that one
RUN curl -L \
    https://github.com/Akashgopalgs/MLOPS-loan-default-prediction/raw/main/src/models/randomforest_best_model.pkl \
    -o /app/src/models/randomforest_best_model.pkl

# Copy the rest of your project (minus the old model)
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Pin joblib to the version you trained with
RUN pip install joblib==1.4.2

# Expose FastAPI port
EXPOSE 8000

# Launch
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
