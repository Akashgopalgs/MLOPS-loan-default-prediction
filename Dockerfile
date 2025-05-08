# Use a base image with Python 3.10
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Ensure that the data folder (with required files) is also copied into the container
COPY src/data/processed /app/src/data/processed

# Train the model (you can specify your script here)
RUN python src/models/train_model.py

# Expose the application port
EXPOSE 8000

# Start FastAPI app when the container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
