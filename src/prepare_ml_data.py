import pandas as pd

input_path = "data/ml/heart_disease/processed.cleveland.data"
output_path = "data/ml/heart_disease_clean.csv"

columns = [
    "Age",
    "Sex",
    "Chest Pain Type",
    "Resting Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Maximum Heart Rate",
    "Exercise Induced Angina",
    "ST Depression",
    "Slope",
    "Major Vessels",
    "Thalassemia",
    "Disease"
]

# Read ? as missing values
df = pd.read_csv(
    input_path,
    header=None,
    names=columns,
    na_values="?"
)

print("Original shape:", df.shape)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Convert target to binary classification
df["Disease"] = (df["Disease"] > 0).astype(int)

# Columns that should be numeric
numeric_columns = [
    "Age",
    "Resting Blood Pressure",
    "Cholesterol",
    "Maximum Heart Rate",
    "ST Depression"
]

# Convert numeric columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Categorical columns
categorical_columns = [
    "Sex",
    "Chest Pain Type",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Exercise Induced Angina",
    "Slope",
    "Major Vessels",
    "Thalassemia"
]

# Convert categorical columns to numeric
for col in categorical_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows where target is missing
df = df.dropna(subset=["Disease"])

# Save dataset with missing values preserved
df.to_csv(output_path, index=False)

print("\nFinal shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDisease distribution:")
print(df["Disease"].value_counts())

print("\nClean dataset saved to:")
print(output_path)