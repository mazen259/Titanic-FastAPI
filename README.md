# 🚢 Titanic Survival Prediction API

A Machine Learning web application that predicts whether a passenger would survive the Titanic disaster using **FastAPI**, **Random Forest Classifier**, and a modern interactive frontend.

---

## 📌 Project Overview

This project uses passenger information such as:

* Passenger Class (`Pclass`)
* Gender (`Sex`)
* Age
* Fare
* Number of Siblings/Spouse (`SibSp`)
* Number of Parents/Children (`Parch`)

Then predicts whether the passenger would:

✅ Survive
❌ Not Survive

---

## 🧠 Machine Learning Model

* Algorithm: **Random Forest Classifier**
* Library: **Scikit-Learn**
* Model saved as:

```text
model.pkl
```

---

## ⚙️ Technologies Used

* Python
* FastAPI
* Pandas
* Scikit-Learn
* Joblib
* HTML / CSS / JavaScript

---

## 📂 Project Structure

```text
Titanic_Project/
│── main.py
│── model.pkl
│── Titanic-Dataset.csv
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run Locally

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Server

```bash
python -m uvicorn main:app --reload
```

### 3️⃣ Open in Browser

```text
http://127.0.0.1:8000
```

### API Docs:

```text
http://127.0.0.1:8000/docs
```

---

## 📥 Example Request

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "Fare": 7.25,
  "SibSp": 0,
  "Parch": 0
}
```

---

## 📤 Example Response

```json
{
  "prediction": "Not Survived",
  "survived_probability": 0.01
}
```

---

## 🌐 Features

✅ Titanic survival prediction
✅ REST API using FastAPI
✅ Interactive Swagger UI
✅ Modern Frontend Website
✅ Probability Output
✅ Input Validation

---

## 📈 Future Improvements

* Deploy online using Render
* Add database for predictions
* Improve UI/UX
* Add charts & analytics
* Use advanced ML models

---

## 👨‍💻 Author

Mazen Maher

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
