import streamlit as st
import pandas as pd

from utils.data_loader import load_hospital_data


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="SmartCare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# Custom Styling
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f766e,
        #0e7490
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.9;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# Load Data
# ==========================================

df = load_hospital_data()


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.title("🏥 SmartCare")

    st.markdown(
        "### Healthcare Data Intelligence"
    )

    st.divider()

    st.info(
        "Explore hospital analytics, "
        "interactive visualizations and "
        "machine learning predictions."
    )

    st.divider()

    st.caption("BDA Project • Healthcare Analytics")


# ==========================================
# Hero Section
# ==========================================

st.markdown("""
<div class="hero">

<h1>🏥 SmartCare</h1>

<p>
Healthcare Data Analytics and Disease Prediction
</p>

<p>
Transforming healthcare data into meaningful insights
through analytics, visualization and machine learning.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# KPI Cards
# ==========================================

total_patients = len(df)
avg_age = df["Age"].mean()
avg_bill = df["Billing Amount"].mean()
avg_stay = df["Length of Stay"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Patient Records",
        f"{total_patients:,}"
    )

with col2:
    st.metric(
        "🎂 Average Age",
        f"{avg_age:.1f} years"
    )

with col3:
    st.metric(
        "💰 Average Billing",
        f"${avg_bill:,.0f}"
    )

with col4:
    st.metric(
        "🛏️ Avg. Length of Stay",
        f"{avg_stay:.1f} days"
    )


# ==========================================
# Overview
# ==========================================

st.markdown(
    '<div class="section-title">📊 Healthcare Overview</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Patients by Medical Condition")

    condition_counts = (
        df["Medical Condition"]
        .value_counts()
        .sort_values(ascending=True)
    )

    st.bar_chart(condition_counts)


with col2:

    st.subheader("Admission Types")

    admission_counts = (
        df["Admission Type"]
        .value_counts()
    )

    st.bar_chart(admission_counts)


# ==========================================
# Project Capabilities
# ==========================================

st.markdown(
    '<div class="section-title">🚀 SmartCare Capabilities</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Hospital Analytics")
    st.write(
        "Analyze patient demographics, "
        "admissions, billing and treatment patterns."
    )

with col2:
    st.markdown("### 📈 Interactive Visualization")
    st.write(
        "Build custom charts by selecting "
        "dimensions, measures and filters."
    )

with col3:
    st.markdown("### 🫀 Disease Prediction")
    st.write(
        "Use a machine learning model to "
        "demonstrate heart disease prediction."
    )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "SmartCare • Big Data Analysis Project • "
    "Healthcare Data Intelligence"
)