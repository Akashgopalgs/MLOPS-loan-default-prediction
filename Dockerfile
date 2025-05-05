# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install curl
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copy your entire project (including the repo's src/models/)
COPY . .

# Now overwrite the model in src/models/ with the one you tested locally
RUN mkdir -p src/models \
 && curl -L \
      https://github.com/Akashgopalgs/MLOPS-loan-default-prediction/raw/main/src/models/randomforest_best_model.pkl \
      -o src/models/randomforest_best_model.pkl

# Upgrade pip & install all dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install joblib==1.4.2

# Expose FastAPI port
EXPOSE 8000

# Launch the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
