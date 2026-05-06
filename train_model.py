import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
df = pd.read_csv(r"pyhthon/Titanic-Dataset.csv")

# Select important features
df = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Survived"]]

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Encode categorical feature
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Split features / target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train / Validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Stronger model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# Save model
joblib.dump(model, "model.pkl")

print("Titanic model saved successfully as model.pkl")
print("Validation Accuracy:", round(acc, 4))