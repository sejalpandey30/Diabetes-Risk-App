import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
}

.big-font {
    font-size: 20px !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==================================================
# FUNCTIONS
# ==================================================

def get_age_group(age):
    if age <= 30:
        return 1
    elif age <= 40:
        return 2
    elif age <= 50:
        return 3
    elif age <= 60:
        return 4
    return 5


def get_bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    return 3

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🩺 Diabetes AI")

    st.markdown("---")

    st.info("""
    AI-powered diabetes risk prediction system.

    Built using:
    - Machine Learning
    - Streamlit
    - Scikit-Learn
    """)

    st.markdown("---")

    st.subheader("Project Highlights")

    st.write("✅ Diabetes Prediction")
    st.write("✅ Risk Analysis")
    st.write("✅ Health Recommendations")
    st.write("✅ Interactive Dashboard")

# ==================================================
# HEADER
# ==================================================

st.title("🩺 AI Diabetes Risk Assessment Dashboard")

st.caption(
    "Predict diabetes risk using machine learning and receive personalized health insights."
)

# ==================================================
# TOP METRICS
# ==================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Model", "Random Forest")

with c2:
    st.metric("Features", "10")

with c3:
    st.metric("Prediction", "Binary Classification")

st.divider()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Recommendations",
    "About Project"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Patient Information")

    left, right = st.columns(2)

    with left:

        pregnancies = st.number_input(
            "Pregnancies",
            0,
            20,
            1
        )

        glucose = st.number_input(
            "Glucose",
            0,
            300,
            120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            0,
            200,
            70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            0,
            100,
            20
        )

    with right:

        insulin = st.number_input(
            "Insulin",
            0,
            900,
            80
        )

        bmi = st.number_input(
            "BMI",
            0.0,
            70.0,
            25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            0.0,
            5.0,
            0.5
        )

        age = st.number_input(
            "Age",
            1,
            100,
            30
        )

    if st.button("🔍 Assess Diabetes Risk", use_container_width=True):

        age_group = get_age_group(age)
        bmi_category = get_bmi_category(bmi)

        patient_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
            age_group,
            bmi_category
        ]])

        patient_scaled = scaler.transform(patient_data)

        prediction = model.predict(patient_scaled)[0]

        probability = model.predict_proba(patient_scaled)[0][1]

        st.divider()

        st.subheader("Assessment Results")

        m1, m2 = st.columns(2)

        with m1:

            st.metric(
                "Risk Probability",
                f"{probability*100:.2f}%"
            )

            st.progress(float(probability))

        with m2:

            if probability < 0.30:
                st.success("🟢 LOW RISK")

            elif probability < 0.70:
                st.warning("🟡 MODERATE RISK")

            else:
                st.error("🔴 HIGH RISK")

        st.markdown("### Risk Visualization")

        fig, ax = plt.subplots(figsize=(6,4))

        categories = [
            "No Diabetes",
            "Diabetes"
        ]

        values = [
            (1-probability)*100,
            probability*100
        ]

        ax.bar(categories, values)

        ax.set_ylabel("Probability (%)")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

        st.session_state["probability"] = probability
        st.session_state["glucose"] = glucose
        st.session_state["bmi"] = bmi
        st.session_state["age"] = age
        st.session_state["bp"] = blood_pressure

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("Personalized Health Recommendations")

    if "probability" not in st.session_state:

        st.info(
            "Run a prediction first to generate recommendations."
        )

    else:

        glucose = st.session_state["glucose"]
        bmi = st.session_state["bmi"]
        age = st.session_state["age"]
        bp = st.session_state["bp"]

        if glucose > 140:
            st.warning(
                "Reduce sugar intake and monitor blood glucose regularly."
            )

        if bmi > 25:
            st.warning(
                "Increase physical activity and maintain healthy body weight."
            )

        if age > 45:
            st.warning(
                "Schedule routine diabetes screening."
            )

        if bp > 130:
            st.warning(
                "Monitor blood pressure frequently."
            )

        if (
            glucose <= 140 and
            bmi <= 25 and
            age <= 45 and
            bp <= 130
        ):
            st.success(
                "Maintain your current healthy lifestyle."
            )

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader("About This Project")

    with st.expander("Project Overview", expanded=True):

        st.write("""
        This AI-powered healthcare dashboard predicts
        diabetes risk using machine learning models
        trained on clinical health indicators.

        Features:
        - Diabetes Prediction
        - Risk Categorization
        - Personalized Recommendations
        - Interactive Dashboard
        """)

    with st.expander("Features Used"):

        st.write("""
        • Pregnancies

        • Glucose

        • Blood Pressure

        • Skin Thickness

        • Insulin

        • BMI

        • Diabetes Pedigree Function

        • Age

        • Age Group

        • BMI Category
        """)

    with st.expander("Technology Stack"):

        st.write("""
        Python

        Streamlit

        Scikit-Learn

        Pandas

        NumPy

        Matplotlib
        """)
       