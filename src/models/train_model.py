import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'processed', 'application_train_featured.csv')
model_output_path = os.path.join(BASE_DIR, 'models')

# Ensure models directory exists
os.makedirs(model_output_path, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
X = df.drop('TARGET', axis=1)
y = df['TARGET']

import pickle

# Save the feature_columns
feature_columns = X.columns.tolist()
feture_col_path = os.path.join(model_output_path, 'feature_columns.pkl')
joblib.dump(feature_columns, feture_col_path)
print(f"💾 Saved feature_columns to: {feture_col_path}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Save the fitted scaler
scaler_path = os.path.join(model_output_path, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"💾 Saved scaler to: {scaler_path}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("✅ Applied SMOTE. Resampled dataset shape:", pd.Series(y_train_resampled).value_counts())

# Define model candidates
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Select best model via cross-validation
best_model_name = None
best_model = None
best_score = 0.0
print("🔍 Comparing models using 5-fold CV on F1-score:")
for name, model in models.items():
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
    mean_score = scores.mean()
    print(f"  {name}: Mean F1 Score = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

# Train the best model on full resampled training data
print(f"\n✅ Best model: {best_model_name} (F1 Score = {best_score:.4f})")
best_model.fit(X_train_resampled, y_train_resampled)

# Evaluate on test data
print("\n📊 Classification Report on Test Set:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the best model
best_model_file = os.path.join(model_output_path, f'{best_model_name.lower()}_best_model.pkl')
joblib.dump(best_model, best_model_file)
print(f"💾 Saved best model to: {best_model_file}")
