# SmartCare 🏥

### Healthcare Data Analytics and Disease Prediction Using Big Data

SmartCare is an interactive healthcare analytics platform that combines data preprocessing, exploratory analysis, interactive visualization, and machine learning-based disease prediction in a single Streamlit application.

The project demonstrates how healthcare data can be transformed into meaningful insights through data analytics and predictive modeling.

---

## 🎯 Project Objective

Healthcare datasets contain valuable information about patients, medical conditions, hospital admissions, billing, medications, test results, and other factors.

SmartCare provides a unified platform to:

- Analyze healthcare admission patterns
- Explore patient demographics and medical conditions
- Visualize billing and length-of-stay patterns
- Build custom interactive visualizations
- Predict heart disease using machine learning
- Evaluate and compare machine learning models

---

## ✨ Features

### 📊 Hospital Analytics

- Patient demographics
- Medical conditions
- Admission types
- Year-wise admission trends
- Billing amounts
- Length of hospital stay
- Billing vs. length of stay

### 🔎 Explore Healthcare Data

- Search patient, doctor, hospital, and medical condition information
- Apply multiple filters
- Inspect selected records
- View dataset statistics
- Download filtered data as CSV

### 📈 Custom Visualization

Users can dynamically select:

- X-axis
- Y-axis
- Aggregation method
- Chart type

Supported charts:

- Bar charts
- Line charts
- Scatter plots
- Box plots

### 🤖 Disease Prediction

SmartCare uses a Random Forest machine learning model to predict whether a patient is likely to have heart disease based on clinical attributes.

### 🧪 Model Performance

The machine learning module provides:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Logistic Regression vs Random Forest comparison

The Random Forest model achieved approximately **90.16% accuracy** on the held-out test set used during development.

---

## 🏗️ System Architecture

```text
                    SMARTCARE
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   Hospital Dataset            Heart Disease Dataset
    54,860 records                  303 records
          │                           │
          ▼                           ▼
 Data Preprocessing            ML Preprocessing
          │                           │
          ▼                           ▼
  Analytics & Visualization    Random Forest Model
          │                           │
          ▼                           ▼
   Streamlit Dashboard          Disease Prediction

```

# 🚀 Run SmartCare Locally

Follow these commands after cloning the repository.

### 1. Clone the repository

```bash
git clone https://github.com/chintam-dhanush/SmartCare.git
```

### 2. Enter the project directory

```bash
cd SmartCare
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the application

```bash
streamlit run app\Home.py
```

### 7. Open the application

After running the command above, open:

```text
http://localhost:8501
```