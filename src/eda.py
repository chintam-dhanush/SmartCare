import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# SMARTCARE - EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

INPUT_FILE = Path("data/clean_healthcare.csv")
OUTPUT_DIR = Path("data/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Convert dates
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

print("=" * 60)
print("SMARTCARE - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nRecords: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# --------------------------------------------------
# 1. Medical Condition Distribution
# --------------------------------------------------

condition_counts = df["Medical Condition"].value_counts()

print("\nMedical Condition Distribution:")
print(condition_counts)

plt.figure(figsize=(9, 5))
sns.barplot(
    x=condition_counts.index,
    y=condition_counts.values
)
plt.title("Patients by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Number of Patients")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "patients_by_condition.png")
plt.close()

# --------------------------------------------------
# 2. Admission Type Distribution
# --------------------------------------------------

admission_counts = df["Admission Type"].value_counts()

print("\nAdmission Type Distribution:")
print(admission_counts)

plt.figure(figsize=(7, 5))
plt.pie(
    admission_counts.values,
    labels=admission_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Admission Type Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "admission_type.png")
plt.close()

# --------------------------------------------------
# 3. Admissions by Year
# --------------------------------------------------

year_counts = df["Admission Year"].value_counts().sort_index()

print("\nAdmissions by Year:")
print(year_counts)

plt.figure(figsize=(9, 5))
sns.lineplot(
    x=year_counts.index,
    y=year_counts.values,
    marker="o"
)
plt.title("Admissions by Year")
plt.xlabel("Year")
plt.ylabel("Number of Admissions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "admissions_by_year.png")
plt.close()

# --------------------------------------------------
# 4. Average Length of Stay by Condition
# --------------------------------------------------

stay_by_condition = (
    df.groupby("Medical Condition")["Length of Stay"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Length of Stay by Condition:")
print(stay_by_condition)

plt.figure(figsize=(9, 5))
sns.barplot(
    x=stay_by_condition.index,
    y=stay_by_condition.values
)
plt.title("Average Length of Stay by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Average Length of Stay (Days)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "stay_by_condition.png")
plt.close()

# --------------------------------------------------
# 5. Average Billing by Condition
# --------------------------------------------------

billing_by_condition = (
    df.groupby("Medical Condition")["Billing Amount"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Billing by Condition:")
print(billing_by_condition)

plt.figure(figsize=(9, 5))
sns.barplot(
    x=billing_by_condition.index,
    y=billing_by_condition.values
)
plt.title("Average Billing Amount by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Average Billing Amount")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "billing_by_condition.png")
plt.close()

# --------------------------------------------------
# 6. Age Distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))
sns.histplot(
    df["Age"],
    bins=20,
    kde=True
)
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.close()

# --------------------------------------------------
# 7. Length of Stay vs Billing
# --------------------------------------------------

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df.sample(min(5000, len(df)), random_state=42),
    x="Length of Stay",
    y="Billing Amount",
    alpha=0.5
)
plt.title("Length of Stay vs Billing Amount")
plt.xlabel("Length of Stay (Days)")
plt.ylabel("Billing Amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "stay_vs_billing.png")
plt.close()

# --------------------------------------------------
# 8. Test Results
# --------------------------------------------------

test_counts = df["Test Results"].value_counts()

print("\nTest Results:")
print(test_counts)

plt.figure(figsize=(7, 5))
sns.barplot(
    x=test_counts.index,
    y=test_counts.values
)
plt.title("Test Result Distribution")
plt.xlabel("Test Result")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_results.png")
plt.close()

# --------------------------------------------------
# 9. Gender Distribution
# --------------------------------------------------

gender_counts = df["Gender"].value_counts()

print("\nGender Distribution:")
print(gender_counts)

# --------------------------------------------------
# 10. Summary Statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("KEY STATISTICS")
print("=" * 60)

print(f"\nAverage Age: {df['Age'].mean():.2f}")
print(f"Average Billing: {df['Billing Amount'].mean():.2f}")
print(f"Average Length of Stay: {df['Length of Stay'].mean():.2f} days")

print(
    f"Most Common Condition: "
    f"{df['Medical Condition'].mode()[0]}"
)

print(
    f"Most Common Admission Type: "
    f"{df['Admission Type'].mode()[0]}"
)

print("\nEDA completed successfully.")

print(f"\nCharts saved to: {OUTPUT_DIR}")