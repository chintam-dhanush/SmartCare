import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_hospital_data


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Custom Visualization",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# Load Data
# ==========================================

df = load_hospital_data()


# ==========================================
# Header
# ==========================================

st.title("📈 Custom Visualization")

st.markdown(
    "Build your own healthcare visualization by selecting "
    "dimensions, measures, aggregation and filters."
)

st.divider()


# ==========================================
# Filters
# ==========================================

st.subheader("🎛️ Data Filters")

col1, col2, col3 = st.columns(3)

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
    admission_type = st.multiselect(
        "Admission Type",
        sorted(df["Admission Type"].unique()),
        default=sorted(df["Admission Type"].unique())
    )


filtered_df = df[
    df["Gender"].isin(gender)
    & df["Medical Condition"].isin(condition)
    & df["Admission Type"].isin(admission_type)
]


# ==========================================
# Age Filter
# ==========================================

age_range = st.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (
        int(df["Age"].min()),
        int(df["Age"].max())
    )
)

filtered_df = filtered_df[
    filtered_df["Age"].between(
        age_range[0],
        age_range[1]
    )
]


st.divider()


# ==========================================
# Visualization Controls
# ==========================================

st.subheader("⚙️ Visualization Builder")

# Columns suitable for grouping
categorical_columns = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
    "Admission Year",
    "Admission Month Name",
    "Admission Weekday"
]

# Numeric columns
numeric_columns = [
    "Age",
    "Billing Amount",
    "Length of Stay",
    "Room Number"
]


col1, col2, col3 = st.columns(3)

with col1:

    x_axis = st.selectbox(
        "X-Axis / Category",
        categorical_columns
    )

with col2:

    y_axis = st.selectbox(
        "Y-Axis / Measure",
        numeric_columns
    )

with col3:

    chart_type = st.selectbox(
        "Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Box Plot"
        ]
    )


aggregation = st.selectbox(
    "Aggregation",
    [
        "Average",
        "Sum",
        "Count",
        "Minimum",
        "Maximum"
    ]
)


# ==========================================
# Generate Button
# ==========================================

generate = st.button(
    "🚀 Generate Visualization",
    type="primary",
    use_container_width=True
)


# ==========================================
# Generate Chart
# ==========================================

if generate:

    if filtered_df.empty:

        st.warning(
            "No data available for the selected filters."
        )

    else:

        # ------------------------------
        # Aggregation
        # ------------------------------

        if aggregation == "Average":

            chart_df = (
                filtered_df
                .groupby(x_axis)[y_axis]
                .mean()
                .reset_index()
            )

            chart_df[y_axis] = chart_df[y_axis].round(2)

        elif aggregation == "Sum":

            chart_df = (
                filtered_df
                .groupby(x_axis)[y_axis]
                .sum()
                .reset_index()
            )

        elif aggregation == "Count":

            chart_df = (
                filtered_df
                .groupby(x_axis)
                .size()
                .reset_index(name="Count")
            )

            y_axis = "Count"

        elif aggregation == "Minimum":

            chart_df = (
                filtered_df
                .groupby(x_axis)[y_axis]
                .min()
                .reset_index()
            )

        else:

            chart_df = (
                filtered_df
                .groupby(x_axis)[y_axis]
                .max()
                .reset_index()
            )


        # ------------------------------
        # Chart
        # ------------------------------

        if chart_type == "Bar Chart":

            fig = px.bar(
                chart_df,
                x=x_axis,
                y=y_axis,
                title=f"{aggregation} of {y_axis} by {x_axis}",
                text_auto=".2s"
            )

        elif chart_type == "Line Chart":

            fig = px.line(
                chart_df,
                x=x_axis,
                y=y_axis,
                markers=True,
                title=f"{aggregation} of {y_axis} by {x_axis}"
            )

        elif chart_type == "Scatter Plot":

            fig = px.scatter(
                chart_df,
                x=x_axis,
                y=y_axis,
                title=f"{aggregation} of {y_axis} by {x_axis}"
            )

        else:

            fig = px.box(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color=x_axis,
                title=f"{y_axis} Distribution by {x_axis}"
            )


        # ------------------------------
        # Chart Layout
        # ------------------------------

        fig.update_layout(
            height=550,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            margin=dict(
                l=40,
                r=40,
                t=80,
                b=40
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ------------------------------
        # Generated Data
        # ------------------------------

        with st.expander("📋 View Chart Data"):

            st.dataframe(
                chart_df,
                use_container_width=True
            )


# ==========================================
# Examples
# ==========================================

st.divider()

st.subheader("💡 Example Analyses")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        **💰 Billing Analysis**

        X → Medical Condition  
        Y → Billing Amount  
        Aggregation → Average  
        Chart → Bar
        """
    )

with col2:

    st.markdown(
        """
        **🛏️ Stay Analysis**

        X → Admission Type  
        Y → Length of Stay  
        Aggregation → Average  
        Chart → Bar
        """
    )

with col3:

    st.markdown(
        """
        **👥 Demographic Analysis**

        X → Medical Condition  
        Y → Age  
        Aggregation → Average  
        Chart → Bar
        """
    )