import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import plotly.express as px


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="🧪",
    layout="wide"
)


# ==========================================
# Load Data + Model
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/ml/heart_disease_clean.csv"
    )


@st.cache_resource
def load_model():
    return joblib.load(
        "models/heart_disease_model.pkl"
    )


df = load_data()
model = load_model()


# ==========================================
# Header
# ==========================================

st.title("🧪 Model Performance")

st.markdown(
    "Evaluation of the Random Forest heart disease "
    "classification model."
)

st.divider()


# ==========================================
# Prepare Data
# ==========================================

X = df.drop("Disease", axis=1)
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


st.subheader("📊 Model Metrics")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


# ==========================================
# Confusion Matrix
# ==========================================

st.divider()

st.subheader("🎯 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Disease", "Actual: Disease"],
    columns=["Predicted: No Disease", "Predicted: Disease"]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    title="Random Forest Confusion Matrix",
    labels={
        "x": "Prediction",
        "y": "Actual",
        "color": "Count"
    }
)

fig_cm.update_layout(
    height=450
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)


# ==========================================
# Model Comparison
# ==========================================

st.divider()

st.subheader("⚖️ Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        0.8689,
        0.9016
    ],
    "Precision": [
        0.8125,
        0.8667
    ],
    "Recall": [
        0.9286,
        0.9286
    ],
    "F1 Score": [
        0.8667,
        0.8966
    ]
})

display_comparison = comparison.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:
    display_comparison[column] = (
        display_comparison[column] * 100
    ).round(2)


st.dataframe(
    display_comparison,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# Comparison Chart
# ==========================================

melted = comparison.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

melted["Score"] = melted["Score"] * 100

fig_comparison = px.bar(
    melted,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    text_auto=".1f",
    title="Model Performance Comparison"
)

fig_comparison.update_layout(
    yaxis_title="Score (%)",
    xaxis_title="Metric",
    yaxis_range=[0, 100],
    height=500
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


# ==========================================
# Interpretation
# ==========================================

st.divider()

st.subheader("💡 Interpretation")

st.markdown(
    f"""
### Random Forest

The Random Forest model achieved an accuracy of
**{accuracy * 100:.2f}%** on the held-out test set.

- **Precision:** {precision * 100:.2f}%
- **Recall:** {recall * 100:.2f}%
- **F1 Score:** {f1 * 100:.2f}%

The model correctly identified most of the positive
heart-disease cases, while also maintaining strong
performance for the no-disease class.

Random Forest performed better overall than the
Logistic Regression baseline in our experiment.
"""
)


# ==========================================
# Dataset Information
# ==========================================

st.divider()

st.subheader("📚 ML Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Records",
        len(df)
    )

with col2:
    st.metric(
        "Training Records",
        len(X_train)
    )

with col3:
    st.metric(
        "Testing Records",
        len(X_test)
    )

st.caption(
    "The evaluation uses a stratified 80/20 train-test split "
    "with a fixed random seed for reproducibility."
)