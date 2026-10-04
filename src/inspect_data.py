import pandas as pd

# Load dataset
df = pd.read_csv("data/healthcare_dataset.csv")

print("=" * 60)
print("SMARTCARE - DATASET INSPECTION")
print("=" * 60)

# Basic information
print("\nDataset Shape:")
print(df.shape)

# Column information
print("\nColumn Information:")
print(df.info())

# Duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Unique values
print("\nUnique Values:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

# Numerical statistics
print("\nNumerical Statistics:")
print(df.describe())

# Categorical distributions
categorical_columns = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Admission Type",
    "Medication",
    "Test Results"
]

for column in categorical_columns:
    print(f"\n{'=' * 40}")
    print(f"{column}")
    print("=" * 40)
    print(df[column].value_counts())

# Date information
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

print("\nAdmission Date Range:")
print(df["Date of Admission"].min(), "to", df["Date of Admission"].max())

# Length of stay
df["Length of Stay"] = (
    df["Discharge Date"] - df["Date of Admission"]
).dt.days

print("\nLength of Stay Statistics:")
print(df["Length of Stay"].describe())

print("\nAnalysis completed successfully.")