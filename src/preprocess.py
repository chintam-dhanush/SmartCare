import pandas as pd
from pathlib import Path

# --------------------------------------------------
# SMARTCARE - DATA PREPROCESSING
# --------------------------------------------------

INPUT_FILE = Path("data/healthcare_dataset.csv")
OUTPUT_FILE = Path("data/clean_healthcare.csv")


def main():

    print("=" * 60)
    print("SMARTCARE - DATA PREPROCESSING")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal records: {len(df):,}")

    # --------------------------------------------------
    # 2. Remove duplicate records
    # --------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate records found: {duplicate_count:,}")

    df = df.drop_duplicates().copy()

    print(f"Records after removing duplicates: {len(df):,}")

    # --------------------------------------------------
    # 3. Convert date columns
    # --------------------------------------------------

    df["Date of Admission"] = pd.to_datetime(
        df["Date of Admission"],
        errors="coerce"
    )

    df["Discharge Date"] = pd.to_datetime(
        df["Discharge Date"],
        errors="coerce"
    )

    # Check invalid dates
    invalid_dates = (
        df["Date of Admission"].isna().sum()
        + df["Discharge Date"].isna().sum()
    )

    print(f"Invalid date values: {invalid_dates}")

    # Remove records with invalid dates
    df = df.dropna(
        subset=["Date of Admission", "Discharge Date"]
    ).copy()

    # --------------------------------------------------
    # 4. Create Length of Stay
    # --------------------------------------------------

    df["Length of Stay"] = (
        df["Discharge Date"] - df["Date of Admission"]
    ).dt.days

    # Remove impossible stay durations
    invalid_stay = (df["Length of Stay"] <= 0).sum()

    print(f"Invalid length-of-stay records: {invalid_stay}")

    df = df[df["Length of Stay"] > 0].copy()

    # --------------------------------------------------
    # 5. Handle negative billing amounts
    # --------------------------------------------------

    negative_billing = (df["Billing Amount"] < 0).sum()

    print(f"Negative billing records: {negative_billing}")

    # Negative billing values are not suitable for
    # cost-based analysis, so remove those records.
    df = df[df["Billing Amount"] >= 0].copy()

    # --------------------------------------------------
    # 6. Create useful date features
    # --------------------------------------------------

    df["Admission Year"] = df["Date of Admission"].dt.year
    df["Admission Month"] = df["Date of Admission"].dt.month
    df["Admission Month Name"] = (
        df["Date of Admission"].dt.month_name()
    )
    df["Admission Day"] = df["Date of Admission"].dt.day
    df["Admission Weekday"] = (
        df["Date of Admission"].dt.day_name()
    )

    # --------------------------------------------------
    # 7. Display final information
    # --------------------------------------------------

    print("\nFinal dataset shape:")
    print(df.shape)

    print("\nFinal columns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nMissing values:")
    print(df.isnull().sum())

    # --------------------------------------------------
    # 8. Save cleaned dataset
    # --------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Clean dataset saved to:")
    print(OUTPUT_FILE)

    print(f"\nFinal records: {len(df):,}")
    print(f"Final columns: {len(df.columns)}")


if __name__ == "__main__":
    main()