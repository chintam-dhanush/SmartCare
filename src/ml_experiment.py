import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------
# SMARTCARE - LEAKAGE-SAFE ML EXPERIMENT
# --------------------------------------------------

DATA_FILE = Path("data/clean_healthcare.csv")

df = pd.read_csv(DATA_FILE)

# --------------------------------------------------
# Date processing
# --------------------------------------------------

df["Date of Admission"] = pd.to_datetime(
    df["Date of Admission"]
)

df["Admission Year"] = (
    df["Date of Admission"].dt.year
)

df["Admission Month"] = (
    df["Date of Admission"].dt.month
)

# --------------------------------------------------
# Create stay category
# --------------------------------------------------

df["Stay Category"] = pd.cut(
    df["Length of Stay"],
    bins=[0, 10, 20, 30],
    labels=["Short", "Medium", "Long"]
)

# --------------------------------------------------
# Candidate features
# --------------------------------------------------

candidate_features = [
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Billing Amount",
    "Admission Type",
    "Medication",
    "Test Results",
    "Admission Year",
    "Admission Month",
]

targets = {
    "Medical Condition": "Medical Condition",
    "Admission Type": "Admission Type",
    "Test Results": "Test Results",
    "Stay Category": "Stay Category",
}


for target_name, target_column in targets.items():

    print("\n" + "=" * 60)
    print(f"TARGET: {target_name}")
    print("=" * 60)

    # --------------------------------------------------
    # Remove target from features
    # --------------------------------------------------

    features = [
        column
        for column in candidate_features
        if column != target_column
    ]

    X = df[features]
    y = df[target_column]

    # --------------------------------------------------
    # Explicitly define categorical columns
    # --------------------------------------------------

    categorical_features = [
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    # Remove target if it happens to be here
    categorical_features = [
        column
        for column in categorical_features
        if column in features
    ]

    numeric_features = [
        column
        for column in features
        if column not in categorical_features
    ]

    print(f"Features used: {len(features)}")

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1:.4f}")