import streamlit as st
import pandas as pd

from utils.data_loader import load_hospital_data


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Explore Data",
    page_icon="🔎",
    layout="wide"
)


# ==========================================
# Load Data
# ==========================================

df = load_hospital_data()


# ==========================================
# Header
# ==========================================

st.title("🔎 Explore Healthcare Data")

st.markdown(
    "Interactively explore, filter and inspect the hospital dataset."
)

st.divider()


# ==========================================
# Search
# ==========================================

st.subheader("🔍 Search Records")

search = st.text_input(
    "Search by patient, hospital, doctor or medical condition",
    placeholder="Example: Diabetes, Apollo, John..."
)


filtered_df = df.copy()

if search:

    search = search.lower()

    mask = (
        filtered_df["Name"].astype(str).str.lower().str.contains(search, na=False)
        | filtered_df["Hospital"].astype(str).str.lower().str.contains(search, na=False)
        | filtered_df["Doctor"].astype(str).str.lower().str.contains(search, na=False)
        | filtered_df["Medical Condition"].astype(str).str.lower().str.contains(search, na=False)
    )

    filtered_df = filtered_df[mask]


# ==========================================
# Filters
# ==========================================

st.subheader("🎛️ Filters")

col1, col2, col3, col4 = st.columns(4)

with col1:

    gender = st.multiselect(
        "Gender",
        sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique())
    )

with col2:

    condition = st.multiselect(
        "Medical Condition",
        sorted(df["Medical Condition"].unique()),
        default=sorted(df["Medical Condition"].unique())
    )

with col3:

    admission = st.multiselect(
        "Admission Type",
        sorted(df["Admission Type"].unique()),
        default=sorted(df["Admission Type"].unique())
    )

with col4:

    test_result = st.multiselect(
        "Test Result",
        sorted(df["Test Results"].unique()),
        default=sorted(df["Test Results"].unique())
    )


filtered_df = filtered_df[
    filtered_df["Gender"].isin(gender)
    & filtered_df["Medical Condition"].isin(condition)
    & filtered_df["Admission Type"].isin(admission)
    & filtered_df["Test Results"].isin(test_result)
]


# ==========================================
# Age Filter
# ==========================================

age_min = int(df["Age"].min())
age_max = int(df["Age"].max())

age_range = st.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

filtered_df = filtered_df[
    filtered_df["Age"].between(
        age_range[0],
        age_range[1]
    )
]


# ==========================================
# Results Summary
# ==========================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Matching Records",
        f"{len(filtered_df):,}"
    )

with col2:
    percentage = (
        len(filtered_df) / len(df) * 100
        if len(df) > 0 else 0
    )

    st.metric(
        "Dataset Coverage",
        f"{percentage:.1f}%"
    )

with col3:
    if len(filtered_df) > 0:
        st.metric(
            "Average Billing",
            f"${filtered_df['Billing Amount'].mean():,.0f}"
        )
    else:
        st.metric(
            "Average Billing",
            "—"
        )


# ==========================================
# Column Selection
# ==========================================

st.subheader("📋 Dataset View")

all_columns = list(df.columns)

default_columns = [
    "Name",
    "Age",
    "Gender",
    "Medical Condition",
    "Hospital",
    "Billing Amount",
    "Admission Type",
    "Medication",
    "Test Results",
    "Length of Stay"
]

selected_columns = st.multiselect(
    "Select columns to display",
    all_columns,
    default=default_columns
)


# ==========================================
# Display Data
# ==========================================

if len(filtered_df) > 0 and selected_columns:

    st.dataframe(
        filtered_df[selected_columns],
        use_container_width=True,
        height=500
    )

else:

    st.warning(
        "No records match the selected filters."
    )


# ==========================================
# Download
# ==========================================

if len(filtered_df) > 0:

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="smartcare_filtered_data.csv",
        mime="text/csv"
    )


# ==========================================
# Dataset Information
# ==========================================

st.divider()

with st.expander("ℹ️ Dataset Information"):

    st.write(
        f"""
        **Total Records:** {len(df):,}

        **Total Columns:** {len(df.columns)}

        **Date Range:** {df["Date of Admission"].min()} 
        to {df["Date of Admission"].max()}

        **Medical Conditions:** {df["Medical Condition"].nunique()}

        **Hospitals:** {df["Hospital"].nunique():,}

        **Doctors:** {df["Doctor"].nunique():,}
        """
    )