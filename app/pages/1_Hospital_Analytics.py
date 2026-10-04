import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_hospital_data


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Hospital Analytics",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# Load Data
# ==========================================

df = load_hospital_data()


# ==========================================
# Page Header
# ==========================================

st.title("📊 Hospital Analytics")
st.markdown(
    "Explore patient demographics, admissions, billing "
    "and treatment patterns."
)

st.divider()


# ==========================================
# Filters
# ==========================================

st.subheader("🔎 Filter Hospital Data")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.multiselect(
        "Gender",
        options=sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique())
    )

with col2:
    conditions = st.multiselect(
        "Medical Condition",
        options=sorted(df["Medical Condition"].unique()),
        default=sorted(df["Medical Condition"].unique())
    )

with col3:
    admission_types = st.multiselect(
        "Admission Type",
        options=sorted(df["Admission Type"].unique()),
        default=sorted(df["Admission Type"].unique())
    )


# ==========================================
# Apply Filters
# ==========================================

filtered_df = df[
    df["Gender"].isin(gender)
    & df["Medical Condition"].isin(conditions)
    & df["Admission Type"].isin(admission_types)
]


# ==========================================
# KPIs
# ==========================================

st.subheader("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patient Records",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Average Age",
        f"{filtered_df['Age'].mean():.1f}"
    )

with col3:
    st.metric(
        "Average Billing",
        f"${filtered_df['Billing Amount'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Average Stay",
        f"{filtered_df['Length of Stay'].mean():.1f} days"
    )


st.divider()


# ==========================================
# Admissions by Year
# ==========================================

year_data = (
    filtered_df
    .groupby("Admission Year")
    .size()
    .reset_index(name="Patients")
)

fig_year = px.line(
    year_data,
    x="Admission Year",
    y="Patients",
    markers=True,
    title="📈 Patient Admissions by Year"
)

fig_year.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Patients"
)


# ==========================================
# Patients by Condition
# ==========================================

condition_data = (
    filtered_df["Medical Condition"]
    .value_counts()
    .reset_index()
)

condition_data.columns = [
    "Medical Condition",
    "Patients"
]

fig_condition = px.bar(
    condition_data,
    x="Medical Condition",
    y="Patients",
    title="🏥 Patients by Medical Condition"
)

fig_condition.update_layout(
    xaxis_title="Medical Condition",
    yaxis_title="Number of Patients"
)


col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig_year,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        fig_condition,
        use_container_width=True
    )


# ==========================================
# Billing Analysis
# ==========================================

billing_data = (
    filtered_df
    .groupby("Medical Condition")["Billing Amount"]
    .mean()
    .reset_index()
    .sort_values("Billing Amount", ascending=False)
)

fig_billing = px.bar(
    billing_data,
    x="Medical Condition",
    y="Billing Amount",
    title="💰 Average Billing by Medical Condition"
)

fig_billing.update_layout(
    xaxis_title="Medical Condition",
    yaxis_title="Average Billing"
)


# ==========================================
# Length of Stay
# ==========================================

stay_data = (
    filtered_df
    .groupby("Medical Condition")["Length of Stay"]
    .mean()
    .reset_index()
    .sort_values("Length of Stay", ascending=False)
)

fig_stay = px.bar(
    stay_data,
    x="Medical Condition",
    y="Length of Stay",
    title="🛏️ Average Length of Stay by Condition"
)

fig_stay.update_layout(
    xaxis_title="Medical Condition",
    yaxis_title="Average Stay (Days)"
)


col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig_billing,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        fig_stay,
        use_container_width=True
    )


# ==========================================
# Billing vs Length of Stay
# ==========================================

st.subheader("💡 Billing vs Length of Stay")

scatter_df = filtered_df.sample(
    min(3000, len(filtered_df)),
    random_state=42
)

fig_scatter = px.scatter(
    scatter_df,
    x="Length of Stay",
    y="Billing Amount",
    color="Medical Condition",
    hover_data=[
        "Age",
        "Gender",
        "Admission Type",
        "Medication"
    ],
    title="Relationship Between Hospital Stay and Billing"
)

fig_scatter.update_layout(
    xaxis_title="Length of Stay (Days)",
    yaxis_title="Billing Amount"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ==========================================
# Dataset Summary
# ==========================================

st.divider()

st.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} hospital records."
)