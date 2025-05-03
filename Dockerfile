# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install curl to download the model file from GitHub
RUN apt-get update && apt-get install -y curl

# Download the model file from GitHub into the /app directory
RUN curl -L https://github.com/Akashgopalgs/MLOPS-loan-default-prediction/raw/main/src/models/randomforest_best_model.pkl \
    -o /app/randomforest_best_model.pkl

# Copy the rest of your project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install your project dependencies
RUN pip install -r requirements.txt

# Pin joblib to the same version you used locally
RUN pip install joblib==1.4.2

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
