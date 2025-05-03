# src/models/predict.py

import sys
from pathlib import Path
import pandas as pd
import joblib

def main():
    # ─── Project paths ────────────────────────────────────────────────────────
    src_root    = Path(__file__).resolve().parents[1]        # loan_default_mlops/src
    model_dir   = src_root / "models"                        # loan_default_mlops/src/models
    data_dir    = src_root / "data" / "processed"            # loan_default_mlops/src/data/processed
    output_dir  = src_root / "data" / "predictions"

    test_data_path = data_dir / "application_test_featured.csv"
    scaler_path    = model_dir / "scaler.pkl"
    model_path     = model_dir / "randomforest_best_model.pkl"
    output_path    = output_dir / "predictions.csv"

    # ─── Sanity checks ───────────────────────────────────────────────────────
    for p in (test_data_path, scaler_path, model_path):
        if not p.exists():
            print(f"❌ File not found: {p}")
            sys.exit(1)

    # ─── Load test data ───────────────────────────────────────────────────────
    df = pd.read_csv(test_data_path)

    # ─── Preserve SK_ID_CURR, keep all feature columns ─────────────────────────
    if "SK_ID_CURR" not in df.columns:
        print("❌ 'SK_ID_CURR' column not found in test data.")
        sys.exit(1)

    ids = df["SK_ID_CURR"]
    X    = df.drop(columns=["TARGET"], errors="ignore")  # keep SK_ID_CURR here

    # ─── Load scaler & model ──────────────────────────────────────────────────
    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    # ─── Reorder columns to match training ────────────────────────────────────
    # scaler.feature_names_in_ holds the exact names seen during fit
    feature_order = list(scaler.feature_names_in_)
    X = X[feature_order]

    # ─── Scale & predict ─────────────────────────────────────────────────────
    X_scaled  = scaler.transform(X)
    preds     = model.predict(X_scaled)
    proba     = model.predict_proba(X_scaled)[:, 1]

    # ─── Save predictions ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET":     preds,
        "PROBABILITY": proba
    })
    out.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()


