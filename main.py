from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Literal

app = FastAPI(title="Titanic Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("model.pkl")


# Request Body
class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=100)
    Fare: float = Field(..., ge=0)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)


# Home Route
@app.get("/")
def home():
    return {"message": "Titanic API is Running"}


# Prediction Route
@app.post("/predict")
def predict(data: Passenger):

    # Encode Sex
    sex = 1 if data.Sex == "female" else 0

    # Feature Engineering
    family_size = data.SibSp + data.Parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Create DataFrame
    df = pd.DataFrame([{
        "Pclass": data.Pclass,
        "Sex": sex,
        "Age": data.Age,
        "Fare": data.Fare,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "FamilySize": family_size,
        "IsAlone": is_alone
    }])

    # Prediction
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])

    return {
        "prediction": "Survived" if pred == 1 else "Not Survived",
        "survived_probability": round(proba, 4)
    }
