from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import pickle
from pathlib import Path
import gzip

app = FastAPI()

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")

# Load model, scaler, and feature columns
BASE = Path(__file__).resolve().parent / "src" / "models"
with open("randomforest_best_model.pkl", "rb") as f_in:
    with gzip.open("randomforest_best_model.pkl.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Step 2: Load the compressed model
with gzip.open("randomforest_best_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

with gzip.open(BASE / "scaler.pkl.gz", "rb") as f:
    scaler = joblib.load(f)

with gzip.open(BASE / "feature_columns.pkl.gz", "rb") as f:
    feature_columns = joblib.load(f)

# Input feature metadata
FEATURES = [
    {"name": "EXT_SOURCE_2", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "EXT_SOURCE_3", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "DAYS_EMPLOYED", "type": "int", "min": -365000, "max": 0},
    {"name": "AMT_INCOME_TOTAL", "type": "float", "min": 0.0, "max": 1000000.0},
    {"name": "AMT_CREDIT", "type": "float", "min": 0.0, "max": 1000000.0},
    {"name": "AMT_ANNUITY", "type": "float", "min": 0.0, "max": 100000.0},
    {"name": "DAYS_REGISTRATION", "type": "int", "min": -25000, "max": 0},
    {"name": "DAYS_ID_PUBLISH", "type": "int", "min": -25000, "max": 0},
    {"name": "CNT_CHILDREN", "type": "int", "min": 0, "max": 20},
    {"name": "NAME_CONTRACT_TYPE", "type": "categorical", "options": ["Cash loans", "Revolving loans"]},
    {"name": "FLAG_OWN_REALTY", "type": "categorical", "options": ["Y", "N"]},
    {"name": "CODE_GENDER", "type": "categorical", "options": ["M", "F"]},
    {"name": "NAME_INCOME_TYPE", "type": "categorical", "options": ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student"]},
    {"name": "OCCUPATION_TYPE", "type": "categorical", "options": ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", "Security staff", "Cleaning staff", "Cooking staff", "Private service staff", "Low-skill Laborers", "Waiters/barmen staff"]}
]

# Set default values for each feature (can be customized further)
DEFAULT_VALUES = {
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "DAYS_EMPLOYED": -10000,
    "AMT_INCOME_TOTAL": 50000.0,
    "AMT_CREDIT": 200000.0,
    "AMT_ANNUITY": 3000.0,
    "DAYS_REGISTRATION": -20000,
    "DAYS_ID_PUBLISH": -20000,
    "CNT_CHILDREN": 0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "FLAG_OWN_REALTY": "Y",
    "CODE_GENDER": "M",
    "NAME_INCOME_TYPE": "Working",
    "OCCUPATION_TYPE": "Laborers"
}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES, "default_values": DEFAULT_VALUES})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    try:
        record = {}
        for feature in FEATURES:
            name = feature["name"]
            value = form.get(name, "").strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")

            if feature["type"] == "float":
                record[name] = float(value)  # Cast to float
            elif feature["type"] == "int":
                record[name] = int(float(value))  # First convert to float, then to int
            else:
                record[name] = value

        df = pd.DataFrame([record])
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(df_encoded)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]

        result = {
            "prediction": int(prediction),
            "probability": f"{probability:.2%}"
        }
        return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES, "result": result, "default_values": DEFAULT_VALUES})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES, "error": str(e), "default_values": DEFAULT_VALUES})
