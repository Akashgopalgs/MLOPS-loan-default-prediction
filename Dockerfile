# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install curl to download the model file from GitHub
RUN apt-get update && apt-get install -y curl

# Download the model file from GitHub into the /app directory
RUN curl -L https://github.com/Akashgopalgs/MLOPS-loan-default-prediction/raw/main/src/models/randomforest_best_model.pkl -o /app/randomforest_best_model.pkl

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI app (adjust path if needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
