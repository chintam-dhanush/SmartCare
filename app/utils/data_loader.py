import pandas as pd
import streamlit as st


@st.cache_data
def load_hospital_data():
    """Load the cleaned hospital dataset."""
    return pd.read_csv("data/clean_healthcare.csv")


@st.cache_data
def load_heart_data():
    """Load the cleaned heart disease dataset."""
    return pd.read_csv("data/ml/heart_disease_clean.csv")