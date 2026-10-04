import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/ml/heart_disease_clean.csv")


# ==========================================
# 2. Separate features and target
# ==========================================

X = df.drop("Disease", axis=1)
y = df["Disease"]


# ==========================================
# 3. Feature types
# ==========================================

categorical_features = [
    "Sex",
    "Chest Pain Type",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Exercise Induced Angina",
    "Slope",
    "Major Vessels",
    "Thalassemia"
]

numeric_features = [
    "Age",
    "Resting Blood Pressure",
    "Cholesterol",
    "Maximum Heart Rate",
    "ST Depression"
]


# ==========================================
# 4. Preprocessing
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 5. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Models
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 7. Train and Evaluate
# ==========================================

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    if name == "Random Forest":
        import joblib

        joblib.dump(
            pipeline,
            "models/heart_disease_model.pkl"
        )

        print("\nRandom Forest model saved successfully!")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# ==========================================
# 8. Model Comparison
# ==========================================

print("\n\nMODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).T

print(results_df.round(4))