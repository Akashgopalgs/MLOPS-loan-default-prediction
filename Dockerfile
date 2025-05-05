# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install curl (for downloading the model) and git (optional, if you want to clone instead)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Download the model file from GitHub
RUN curl -L \
    https://github.com/Akashgopalgs/MLOPS-loan-default-prediction/raw/main/src/models/randomforest_best_model.pkl \
    -o /app/randomforest_best_model.pkl

# Copy the rest of your project
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install your requirements, make sure 'joblib' is pinned in requirements.txt
RUN pip install -r requirements.txt

# **Pin joblib** to exactly the version you trained with
RUN pip install joblib==1.4.2

# Expose FastAPI port
EXPOSE 8000

# Launch the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
