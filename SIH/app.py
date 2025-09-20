import os
import pandas as pd
import joblib
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------
# Load data and models
# -----------------------------
df = pd.read_csv("rainfall/Odisha_Rainfall.csv")

load_dotenv("api.env")
WEATHER_KEY = os.getenv("WEATHER_API")

fertility_model = joblib.load("models/soil_fertility_rf.pkl")
imputer = joblib.load("models/imputer.pkl")
yield_model = joblib.load("yield prediction/crop_yield_rf.pkl")

# -----------------------------
# Create app
# -----------------------------
app = FastAPI()

# -----------------------------
# Month mapping
# -----------------------------
MONTH_MAP = {
    "january": "Jan", "jan": "Jan",
    "february": "Feb", "feb": "Feb",
    "march": "Mar", "mar": "Mar",
    "april": "Apr", "apr": "Apr",
    "may": "May",
    "june": "Jun", "jun": "Jun",
    "july": "Jul", "jul": "Jul",
    "august": "Aug", "aug": "Aug",
    "september": "Sep", "sep": "Sep",
    "october": "Oct", "oct": "Oct",
    "november": "Nov", "nov": "Nov",
    "december": "Dec", "dec": "Dec",
}

def classify_rainfall(value: float) -> str:
    if value < 50:
        return "Low"
    elif value <= 200:
        return "Medium"
    else:
        return "Heavy"

# -----------------------------
# Rainfall Endpoint
# -----------------------------
@app.get("/rainfall/")
def get_rainfall(district: str, month: str):
    try:
        if month.lower() in MONTH_MAP:
            month = MONTH_MAP[month.lower()]
        else:
            return {"error": "Invalid month"}

        row = df[df["District"].str.lower() == district.lower()]
        if row.empty:
            return {"error": "District not found"}
        
        rainfall = float(row[month].values[0])
        category = classify_rainfall(rainfall)
        
        return {
            "district": district,
            "month": month,
            "rainfall_mm": rainfall,
            "category": category
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Soil Fertility Endpoint
# -----------------------------
class SoilData(BaseModel):
    N: float
    P: float
    K: float
    ph: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float

@app.post("/predict_fertility")
def predict_fertility(data: SoilData):
    features = [[
        data.N, data.P, data.K, data.ph, data.EC, data.OC,
        data.S, data.Zn, data.Fe, data.Cu, data.Mn, data.B
    ]]
    features = imputer.transform(features)
    prediction = fertility_model.predict(features)[0]
    return {"fertility_score": float(prediction)}

# -----------------------------
# Weather API Endpoint
# -----------------------------
class Location(BaseModel):
    location: str

@app.post("/weather")
def get_weather_api(data: Location):
    url = "http://api.weatherstack.com/current"
    params = {
        "access_key": WEATHER_KEY,
        "query": data.location,
        "units": "m"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "error" in data:
        return {"error": data["error"]["info"]}
    
    return {
        "location": data["location"]["name"],
        "country": data["location"]["country"],
        "temperature": data["current"]["temperature"],
        "humidity": data["current"]["humidity"],
        "condition": ", ".join(data["current"]["weather_descriptions"])
    }

# -----------------------------
# Crop Yield Endpoint
# -----------------------------
class YieldData(BaseModel):
    CROP: str
    SEASON: str
    STATE: str
    AREA: float
    ANNUAL_RAINFALL: float
    FERTILIZER: float
    PESTICIDE: float

@app.post("/predict_yield")
def predict_yield(data: YieldData):
    df_input = pd.DataFrame([data.dict()])

    # One-hot encode categorical features to match training
    df_input = pd.get_dummies(df_input)

    # Ensure all training columns exist
    for col in yield_model.feature_names_in_:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder columns
    df_input = df_input[yield_model.feature_names_in_]

    # Predict yield
    prediction = yield_model.predict(df_input)[0]

    # ✅ Friendly explanation
    explanation = (
        f"For {data.CROP} in {data.SEASON} season at {data.STATE}, "
        f"with an area of {data.AREA} hectares, "
        f"annual rainfall {data.ANNUAL_RAINFALL} mm, "
        f"fertilizer {data.FERTILIZER} kg/ha and pesticide {data.PESTICIDE} kg/ha, "
        f"the predicted yield is **{prediction:.2f} tons/hectare**."
    )

    return {
        "predicted_yield": float(prediction),
        "message": explanation
    }
