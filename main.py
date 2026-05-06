from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Literal

app = FastAPI(title="Titanic Prediction API")

# Allow requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = joblib.load("model.pkl")


# Request Body
class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=100)
    Fare: float = Field(..., ge=0)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)


# Home Page (Website)
@app.get("/", response_class=HTMLResponse)
def website():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic Predictor</title>
        <style>
            body{
                font-family:Arial;
                background:#0f172a;
                color:white;
                display:flex;
                justify-content:center;
                align-items:center;
                min-height:100vh;
            }
            .box{
                background:#1e293b;
                padding:30px;
                border-radius:20px;
                width:400px;
            }
            input,select,button{
                width:100%;
                padding:10px;
                margin:8px 0;
                border:none;
                border-radius:10px;
            }
            button{
                background:#38bdf8;
                font-weight:bold;
            }
            #result{
                margin-top:15px;
                padding:10px;
                background:#334155;
                border-radius:10px;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>🚢 Titanic Predictor</h2>

            <select id="Pclass">
                <option value="1">First Class</option>
                <option value="2">Second Class</option>
                <option value="3">Third Class</option>
            </select>

            <select id="Sex">
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>

            <input id="Age" type="number" placeholder="Age">
            <input id="Fare" type="number" placeholder="Fare">
            <input id="SibSp" type="number" placeholder="Siblings/Spouse">
            <input id="Parch" type="number" placeholder="Parents/Children">

            <button onclick="predict()">Predict</button>

            <div id="result">Waiting...</div>
        </div>

        <script>
        async function predict(){
            const data = {
                Pclass:+document.getElementById("Pclass").value,
                Sex:document.getElementById("Sex").value,
                Age:+document.getElementById("Age").value,
                Fare:+document.getElementById("Fare").value,
                SibSp:+document.getElementById("SibSp").value,
                Parch:+document.getElementById("Parch").value
            };

            const res = await fetch('/predict',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify(data)
            });

            const json = await res.json();

            document.getElementById("result").innerHTML =
                "Prediction: " + json.prediction +
                "<br>Probability: " + json.survived_probability;
        }
        </script>
    </body>
    </html>
    """


# Prediction API
@app.post("/predict")
def predict(data: Passenger):

    sex = 1 if data.Sex == "female" else 0
    family_size = data.SibSp + data.Parch + 1
    is_alone = 1 if family_size == 1 else 0

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

    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])

    return {
        "prediction": "Survived" if pred == 1 else "Not Survived",
        "survived_probability": round(proba, 4)
    }