# # loan_default_mlops/app.py
#
# from fastapi import FastAPI, HTTPException
# import pandas as pd
# import joblib
# from pathlib import Path
#
# app = FastAPI()
#
# # ─── PATH SETUP ──────────────────────────────────────────────────────────
# BASE_DIR    = Path(__file__).resolve().parent
# model_path  = BASE_DIR / "src" / "models" / "randomforest_best_model.pkl"
# scaler_path = BASE_DIR / "src" / "models" / "scaler.pkl"
#
# print("Model path:",  model_path)
# print("Scaler path:", scaler_path)
# if not model_path.exists():
#     raise FileNotFoundError(f"Model not found at {model_path}")
# if not scaler_path.exists():
#     raise FileNotFoundError(f"Scaler not found at {scaler_path}")
#
# model  = joblib.load(model_path)
# scaler = joblib.load(scaler_path)
# # ────────────────────────────────────────────────────────────────────────
#
#
# @app.get("/")
# def read_root():
#     return {"message": "Loan Default Prediction API is online."}
#
#
# @app.post("/predict")
# def predict(data: list[dict]):
#     # 1) Build DataFrame
#     df = pd.DataFrame(data)
#
#     # 2) Ensure all features are present
#     feature_order = list(scaler.feature_names_in_)
#     missing       = set(feature_order) - set(df.columns)
#     if missing:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Missing features: {', '.join(sorted(missing))}"
#         )
#
#     # 3) All inputs must be numeric—coerce and check
#     try:
#         df = df[feature_order].astype(float)
#     except ValueError as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Non-numeric data encountered: {e}"
#         )
#
#     # 4) Scale & predict
#     X_scaled = scaler.transform(df)
#     preds    = model.predict(X_scaled)
#     proba    = model.predict_proba(X_scaled)[:, 1]
#
#     return {"predictions": preds.tolist(), "probabilities": proba.tolist()}


# loan_default_mlops/app.py
# loan_default_mlops/app.py

# loan_default_mlops/app.py

import os
import joblib
import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI()

# ─── PATHS ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent / "src" / "models"
MODEL_PATH    = BASE / "randomforest_best_model.pkl"
SCALER_PATH   = BASE / "scaler.pkl"
FEATURES_PATH = BASE / "feature_columns.pkl"
# ────────────────────────────────────────────────────────────────────────

# ─── LOAD ARTIFACTS ─────────────────────────────────────────────────────
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
feature_columns = pickle.load(open(FEATURES_PATH, "rb"))
# ────────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Loan Default Prediction API is live."}

@app.post("/predict")
def predict(record: dict):
    """
    Accepts a JSON object (record), one-hot–encodes it,
    aligns it to the training columns, scales, and predicts.
    """
    try:
        # 1) Raw JSON → DataFrame
        df = pd.DataFrame([record])

        # 2) One-hot encode (must match training preprocessing)
        df_enc = pd.get_dummies(df)

        # 3) Align to exactly the training columns; fill missing with 0
        df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)

        # 4) Scale
        X_scaled = scaler.transform(df_enc)

        # 5) Predict
        pred  = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0, 1]

        return {"prediction": int(pred), "probability": float(proba)}

    except Exception as e:
        # any issue—bad payload, missing libs, etc.
        raise HTTPException(status_code=400, detail=str(e))
