import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🫀",
    layout="wide"
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/heart_disease_model.pkl"
    )


model = load_model()


# ==========================================
# Header
# ==========================================

st.title("🫀 Heart Disease Prediction")

st.markdown(
    "Enter patient parameters to generate a machine learning "
    "prediction using the trained Random Forest model."
)

st.info(
    "⚠️ Educational demonstration only. "
    "This prediction is based on a machine learning dataset "
    "and is not a medical diagnosis."
)

st.divider()


# ==========================================
# Patient Information
# ==========================================

st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

with col2:

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
            "Female" if x == 0 else "Male"
    )

with col3:

    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Type {x}"
    )


# ==========================================
# Clinical Measurements
# ==========================================

st.subheader("🩺 Clinical Measurements")

col1, col2, col3 = st.columns(3)

with col1:

    blood_pressure = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=130
    )

with col2:

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=600,
        value=220
    )

with col3:

    max_heart_rate = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )


# ==========================================
# Test / ECG Information
# ==========================================

st.subheader("🧪 Test & ECG Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        options=[0, 1],
        format_func=lambda x:
            "≤ 120 mg/dl" if x == 0
            else "> 120 mg/dl"
    )

with col2:

    resting_ecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2],
        format_func=lambda x: f"Type {x}"
    )

with col3:

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col4:

    st_depression = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


# ==========================================
# Additional Features
# ==========================================

st.subheader("📋 Additional Clinical Features")

col1, col2, col3 = st.columns(3)

with col1:

    slope = st.selectbox(
        "Slope",
        options=[1, 2, 3]
    )

with col2:

    major_vessels = st.selectbox(
        "Major Vessels",
        options=[0, 1, 2, 3]
    )

with col3:

    thalassemia = st.selectbox(
        "Thalassemia",
        options=[3, 6, 7]
    )


st.divider()


# ==========================================
# Prediction
# ==========================================

predict = st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
)


if predict:

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Chest Pain Type": chest_pain,
        "Resting Blood Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Fasting Blood Sugar": fasting_bs,
        "Resting ECG": resting_ecg,
        "Maximum Heart Rate": max_heart_rate,
        "Exercise Induced Angina": exercise_angina,
        "ST Depression": st_depression,
        "Slope": slope,
        "Major Vessels": major_vessels,
        "Thalassemia": thalassemia
    }])


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    st.divider()

    st.subheader("📊 Prediction Result")


    col1, col2 = st.columns(2)


    with col1:

        if prediction == 1:

            st.error(
                "🫀 Prediction: Heart Disease"
            )

        else:

            st.success(
                "✅ Prediction: No Heart Disease"
            )


    with col2:

        st.metric(
            "Disease Probability",
            f"{probability * 100:.1f}%"
        )


    st.progress(
        float(probability)
    )


    if probability >= 0.5:

        st.warning(
            "The model predicts a higher likelihood "
            "of heart disease for the provided inputs."
        )

    else:

        st.info(
            "The model predicts a lower likelihood "
            "of heart disease for the provided inputs."
        )


    with st.expander("🔎 View Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )