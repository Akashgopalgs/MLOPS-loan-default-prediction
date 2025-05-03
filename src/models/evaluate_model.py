import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the cleaned data
data_path =  r"C:\Users\akash\OneDrive\Documents\MLOPS\End_to_end_MLOPS_project\loan_default_mlops\src\data\processed\application_train_featured.csv"

 # Adjusted path
data = pd.read_csv(data_path)

# Features and target
X = data.drop("TARGET", axis=1)
y = data["TARGET"]

# Split into train and test sets (same as train_model.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data (same scaler as training)
scaler_path = "scaler.pkl"  # Adjusted path
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# Load the best model
model_path = "randomforest_best_model.pkl"  # Adjusted path
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # For ROC AUC

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Model Evaluation Results:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# (Optional) Save to file
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}
metrics_path = "../../src/models/model_metrics.json"  # Adjusted path
pd.Series(metrics).to_json(metrics_path)

print(f"\nMetrics saved to: {metrics_path}")
