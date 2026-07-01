import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# ==================================================
# PAGE CONFIG & THEME
# ==================================================

st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ==================================================

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        padding-top: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        border: none;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card-light {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
    }
    
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        color: #155724;
    }
    
    .risk-moderate {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 10px;
        color: #856404;
    }
    
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 10px;
        color: #721c24;
    }
    
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 28px;
        font-weight: 700;
        margin: 20px 0 10px 0;
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# UTILITY FUNCTIONS
# ==================================================

@st.cache_resource
def load_model_and_scaler():
    """Load ML model and scaler with error handling"""
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure diabetes_model.pkl and scaler.pkl exist.")
        st.stop()

def get_age_group(age):
    """Categorize age into groups"""
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
    """Categorize BMI"""
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    return 3

def get_risk_level(probability):
    """Determine risk level and details"""
    if probability < 0.30:
        return "🟢 LOW RISK", "low", probability
    elif probability < 0.70:
        return "🟡 MODERATE RISK", "moderate", probability
    else:
        return "🔴 HIGH RISK", "high", probability

def get_health_recommendations(glucose, bmi, age, bp, insulin, dpf):
    """Generate personalized health recommendations based on metrics"""
    recommendations = []
    risk_factors = []
    
    # Glucose recommendations
    if glucose > 140:
        risk_factors.append("High Glucose")
        recommendations.append({
            "category": "🍎 Nutrition",
            "advice": "Reduce sugar and refined carbohydrate intake. Focus on whole grains and fiber-rich foods.",
            "priority": "HIGH"
        })
    else:
        recommendations.append({
            "category": "🍎 Nutrition",
            "advice": "Maintain balanced diet with regular meal timing.",
            "priority": "LOW"
        })
    
    # BMI recommendations
    if bmi > 25:
        risk_factors.append("Overweight/Obesity")
        recommendations.append({
            "category": "💪 Exercise",
            "advice": f"Increase physical activity to 150 min/week. Target BMI: 18.5-24.9. Current BMI: {bmi:.1f}",
            "priority": "HIGH"
        })
    else:
        recommendations.append({
            "category": "💪 Exercise",
            "advice": "Maintain regular physical activity of 150 min/week.",
            "priority": "LOW"
        })
    
    # Age recommendations
    if age > 45:
        risk_factors.append("Age > 45")
        recommendations.append({
            "category": "🏥 Screening",
            "advice": "Schedule annual diabetes screening and health check-ups.",
            "priority": "HIGH"
        })
    else:
        recommendations.append({
            "category": "🏥 Screening",
            "advice": "Maintain regular health check-ups (every 2 years).",
            "priority": "LOW"
        })
    
    # Blood Pressure recommendations
    if bp > 130:
        risk_factors.append("High Blood Pressure")
        recommendations.append({
            "category": "❤️ Cardiovascular",
            "advice": "Monitor blood pressure frequently. Reduce sodium intake and manage stress.",
            "priority": "HIGH"
        })
    else:
        recommendations.append({
            "category": "❤️ Cardiovascular",
            "advice": "Maintain current BP management routine.",
            "priority": "LOW"
        })
    
    # Insulin recommendations
    if insulin > 250:
        risk_factors.append("High Insulin Levels")
        recommendations.append({
            "category": "🧬 Metabolic",
            "advice": "Consult endocrinologist. May indicate insulin resistance.",
            "priority": "HIGH"
        })
    
    # DPF recommendations
    if dpf > 1.0:
        risk_factors.append("Strong Family History")
        recommendations.append({
            "category": "👨‍👩‍👧‍👦 Family History",
            "advice": "Higher genetic predisposition. Extra vigilance in lifestyle management recommended.",
            "priority": "MEDIUM"
        })
    
    return recommendations, risk_factors

def save_prediction_to_history(prediction_data):
    """Save prediction to session history"""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_history.append(prediction_data)

def export_to_csv():
    """Export prediction history to CSV"""
    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        return df.to_csv(index=False).encode('utf-8')
    return None

# ==================================================
# LOAD MODEL
# ==================================================

model, scaler = load_model_and_scaler()

# ==================================================
# SIDEBAR - NAVIGATION & INFO
# ==================================================

with st.sidebar:
    st.title("🩺 Diabetes AI")
    st.markdown("---")
    
    st.info("""
    ### 🤖 About This App
    
    AI-powered diabetes risk prediction system using machine learning.
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Project Features")
    st.write("""
    ✅ Diabetes Risk Prediction
    ✅ Risk Stratification
    ✅ Personalized Recommendations
    ✅ Health Metrics Analysis
    ✅ Prediction History
    ✅ Data Export
    ✅ Feature Insights
    """)
    
    st.markdown("---")
    
    st.subheader("🛠️ Technology Stack")
    cols = st.columns(2)
    with cols[0]:
        st.write("• Python")
        st.write("• Streamlit")
        st.write("• Scikit-Learn")
    with cols[1]:
        st.write("• Pandas")
        st.write("• NumPy")
        st.write("• Matplotlib")
    
    st.markdown("---")
    
    st.subheader("⚠️ Disclaimer")
    st.caption(
        "This tool provides estimates only and should not replace professional medical advice. "
        "Always consult with healthcare professionals for diagnosis and treatment."
    )

# ==================================================
# MAIN HEADER
# ==================================================

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown('<div class="section-header">🩺 AI Diabetes Risk Assessment Dashboard</div>', unsafe_allow_html=True)
with col2:
    st.write("")  # Spacing

st.markdown(
    "💡 Predict diabetes risk using advanced machine learning and receive data-driven health insights."
)

# ==================================================
# TOP METRICS CARDS
# ==================================================

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("🤖 Model Type", "Random Forest")

with metric_col2:
    st.metric("📊 Input Features", "10")

with metric_col3:
    st.metric("🎯 Prediction Type", "Binary Classification")

with metric_col4:
    st.metric("📈 Accuracy Focus", "Risk Stratification")

st.divider()

# ==================================================
# MAIN TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Prediction",
    "📋 Recommendations",
    "📊 Analytics",
    "📁 History",
    "ℹ️ About"
])

# ==================================================
# TAB 1: PREDICTION
# ==================================================

with tab1:
    st.markdown('<div class="section-header">Patient Health Assessment</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    📝 <b>Instructions:</b> Enter your health metrics below. All fields are required for accurate prediction.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section with better UX
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Metrics")
        
        pregnancies = st.number_input(
            "👶 Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Number of times pregnant"
        )
        
        glucose = st.number_input(
            "🍬 Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
            help="Fasting blood glucose level. Normal: <100, Prediabetic: 100-125, Diabetic: ≥126"
        )
        
        blood_pressure = st.number_input(
            "💓 Blood Pressure (mmHg)",
            min_value=0,
            max_value=200,
            value=70,
            help="Diastolic blood pressure. Normal: <80, High: ≥140"
        )
        
        skin_thickness = st.number_input(
            "📏 Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        st.subheader("Advanced Metrics")
        
        insulin = st.number_input(
            "💧 Insulin (mIU/mL)",
            min_value=0,
            max_value=900,
            value=80,
            help="2-Hour serum insulin level. Normal: <12, High: >150"
        )
        
        bmi = st.number_input(
            "⚖️ BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index. Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30"
        )
        
        dpf = st.number_input(
            "👨‍👩‍👧‍👦 Diabetes Pedigree Function",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Diabetes history in family. Higher = stronger family history"
        )
        
        age = st.number_input(
            "🎂 Age (years)",
            min_value=1,
            max_value=100,
            value=30,
            help="Your age in years"
        )
    
    st.divider()
    
    # Prediction Button
    col_btn1, col_btn2 = st.columns([0.6, 0.4])
    
    with col_btn1:
        predict_button = st.button(
            "🔍 Assess Diabetes Risk",
            use_container_width=True,
            help="Click to generate diabetes risk prediction"
        )
    
    with col_btn2:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.pop("probability", None)
            st.session_state.pop("prediction_data", None)
            st.rerun()
    
    # Prediction Logic
    if predict_button:
        # Data preparation
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
        
        # Model prediction
        try:
            patient_scaled = scaler.transform(patient_data)
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0][1]
            
            # Store in session
            st.session_state["probability"] = probability
            st.session_state["prediction_data"] = {
                "pregnancies": pregnancies,
                "glucose": glucose,
                "blood_pressure": blood_pressure,
                "skin_thickness": skin_thickness,
                "insulin": insulin,
                "bmi": bmi,
                "dpf": dpf,
                "age": age,
                "risk_probability": probability
            }
            
            # Save to history
            save_prediction_to_history(st.session_state["prediction_data"])
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()
    
    # Display Results
    if "probability" in st.session_state:
        st.divider()
        st.markdown('<div class="section-header">📊 Assessment Results</div>', unsafe_allow_html=True)
        
        probability = st.session_state["probability"]
        risk_label, risk_level, _ = get_risk_level(probability)
        
        # Risk Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Probability",
                f"{probability*100:.2f}%",
                delta="Updated just now"
            )
        
        with col2:
            confidence = (1 - abs(probability - 0.5) * 2) * 100
            st.metric(
                "Prediction Confidence",
                f"{confidence:.1f}%"
            )
        
        with col3:
            if risk_level == "low":
                st.metric("Risk Category", "🟢 LOW")
            elif risk_level == "moderate":
                st.metric("Risk Category", "🟡 MODERATE")
            else:
                st.metric("Risk Category", "🔴 HIGH")
        
        st.divider()
        
        # Risk Display Box
        st.markdown(f'<div class="risk-{risk_level}"><b>{risk_label}</b></div>', unsafe_allow_html=True)
        
        # Risk Visualization
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("### Risk Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            categories = ["No Diabetes", "Diabetes"]
            values = [(1-probability)*100, probability*100]
            colors = ["#28a745", "#dc3545"]
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
            ax.set_ylabel("Probability (%)", fontsize=11, fontweight="bold")
            ax.set_title("Prediction Confidence Distribution", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_viz2:
            st.markdown("### Health Metrics Overview")
            
            metrics_data = {
                "Glucose": glucose,
                "BMI": bmi,
                "Blood Pressure": blood_pressure,
                "Insulin": insulin,
                "Age": age
            }
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Normalize for radar-like visualization
            normalized_vals = [
                min(glucose/300 * 100, 100),
                min(bmi/30 * 100, 100),
                min(blood_pressure/200 * 100, 100),
                min(insulin/900 * 100, 100),
                min(age/100 * 100, 100)
            ]
            
            bars = ax.barh(list(metrics_data.keys()), normalized_vals, color="#667eea", alpha=0.7)
            ax.set_xlabel("Relative Level (%)", fontweight="bold")
            ax.set_title("Normalized Health Metrics", fontweight="bold")
            ax.set_xlim(0, 100)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.0f}%', ha='left', va='center', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================================================
# TAB 2: RECOMMENDATIONS
# ==================================================

with tab2:
    st.markdown('<div class="section-header">💊 Personalized Health Recommendations</div>', unsafe_allow_html=True)
    
    if "prediction_data" not in st.session_state:
        st.markdown("""
        <div class="info-box">
        ℹ️ <b>No active prediction:</b> Run a prediction first in the "Prediction" tab to generate personalized recommendations.
        </div>
        """, unsafe_allow_html=True)
    else:
        pred_data = st.session_state["prediction_data"]
        recommendations, risk_factors = get_health_recommendations(
            pred_data["glucose"],
            pred_data["bmi"],
            pred_data["age"],
            pred_data["blood_pressure"],
            pred_data["insulin"],
            pred_data["dpf"]
        )
        
        # Risk Factors Summary
        if risk_factors:
            st.markdown("### ⚠️ Identified Risk Factors")
            risk_cols = st.columns(len(risk_factors)) if risk_factors else [st.columns(1)[0]]
            for idx, factor in enumerate(risk_factors):
                with st.columns(len(risk_factors))[idx]:
                    st.warning(f"🚨 {factor}")
        else:
            st.success("✅ No significant risk factors detected!")
        
        st.divider()
        
        # Recommendations by Priority
        st.markdown("### 📋 Actionable Recommendations")
        
        high_priority = [r for r in recommendations if r["priority"] == "HIGH"]
        medium_priority = [r for r in recommendations if r["priority"] == "MEDIUM"]
        low_priority = [r for r in recommendations if r["priority"] == "LOW"]
        
        if high_priority:
            st.markdown("#### 🔴 High Priority")
            for rec in high_priority:
                st.markdown(f"""
                **{rec["category"]}**
                {rec["advice"]}
                """)
        
        if medium_priority:
            st.markdown("#### 🟡 Medium Priority")
            for rec in medium_priority:
                st.markdown(f"""
                **{rec["category"]}**
                {rec["advice"]}
                """)
        
        if low_priority:
            with st.expander("ℹ️ General Recommendations (Low Priority)"):
                for rec in low_priority:
                    st.markdown(f"""
                    **{rec["category"]}**
                    {rec["advice"]}
                    """)
        
        st.divider()
        
        # Additional Health Guidelines
        st.markdown("### 📖 General Diabetes Prevention Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🥗 Nutrition:**
            - Eat whole grains, lean proteins, vegetables
            - Limit sugar and refined carbs
            - Portion control
            - Regular meal timing
            
            **💪 Exercise:**
            - Aim for 150 min moderate activity/week
            - Strength training 2-3x/week
            - Reduce sedentary time
            """)
        
        with col2:
            st.markdown("""
            **😴 Lifestyle:**
            - 7-9 hours quality sleep
            - Stress management
            - Limit alcohol
            - Avoid smoking
            
            **🏥 Monitoring:**
            - Regular health check-ups
            - Annual glucose screening (if 45+)
            - Track vital signs
            """)

# ==================================================
# TAB 3: ANALYTICS
# ==================================================

with tab3:
    st.markdown('<div class="section-header">📊 Detailed Analytics & Insights</div>', unsafe_allow_html=True)
    
    if "prediction_data" not in st.session_state:
        st.info("Run a prediction first to view analytics.")
    else:
        pred_data = st.session_state["prediction_data"]
        probability = st.session_state["probability"]
        
        # Feature Contribution Analysis
        st.markdown("### 🔍 Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Your Input Metrics")
            metrics_display = pd.DataFrame({
                "Metric": ["Glucose", "BMI", "Blood Pressure", "Insulin", "Age", "Family History"],
                "Value": [
                    f"{pred_data['glucose']} mg/dL",
                    f"{pred_data['bmi']:.1f} kg/m²",
                    f"{pred_data['blood_pressure']} mmHg",
                    f"{pred_data['insulin']} mIU/mL",
                    f"{pred_data['age']} years",
                    f"{pred_data['dpf']:.2f}"
                ],
                "Status": [
                    "🟢 Normal" if pred_data['glucose'] < 100 else "🟡 Elevated" if pred_data['glucose'] < 126 else "🔴 High",
                    "🟢 Normal" if pred_data['bmi'] < 25 else "🟡 Overweight" if pred_data['bmi'] < 30 else "🔴 Obese",
                    "🟢 Normal" if pred_data['blood_pressure'] < 120 else "🟡 Elevated" if pred_data['blood_pressure'] < 140 else "🔴 High",
                    "🟢 Normal" if pred_data['insulin'] < 12 else "🟡 Elevated" if pred_data['insulin'] < 150 else "🔴 High",
                    "🟢 <45" if pred_data['age'] < 45 else "🟡 45-60" if pred_data['age'] < 60 else "🔴 >60",
                    "🟢 Low" if pred_data['dpf'] < 0.5 else "🟡 Medium" if pred_data['dpf'] < 1.0 else "🔴 High"
                ]
            })
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Risk Level Interpretation")
            
            if probability < 0.30:
                st.success("""
                ### 🟢 LOW RISK
                
                **Your diabetes risk is relatively low.**
                
                - Continue current healthy lifestyle
                - Annual health check-ups
                - Maintain regular exercise
                - Monitor diet
                """)
            elif probability < 0.70:
                st.warning("""
                ### 🟡 MODERATE RISK
                
                **Your diabetes risk is elevated.**
                
                - Implement lifestyle changes
                - More frequent monitoring
                - Consult healthcare provider
                - Focus on identified risk factors
                """)
            else:
                st.error("""
                ### 🔴 HIGH RISK
                
                **Your diabetes risk is significant.**
                
                - ⚠️ Consult healthcare provider urgently
                - Consider diabetes screening
                - Aggressive lifestyle modification
                - Regular monitoring essential
                """)
        
        st.divider()
        
        # Risk Trend (if multiple predictions exist)
        if len(st.session_state.get("prediction_history", [])) > 1:
            st.markdown("### 📈 Prediction History Trend")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history_df['timestamp'], history_df['risk_probability'] * 100, marker='o', linewidth=2, markersize=8)
            ax.fill_between(history_df['timestamp'], history_df['risk_probability'] * 100, alpha=0.3)
            ax.set_ylabel("Risk Probability (%)", fontweight="bold")
            ax.set_xlabel("Prediction Date", fontweight="bold")
            ax.set_title("Your Diabetes Risk Trend Over Time", fontweight="bold")
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ==================================================
# TAB 4: PREDICTION HISTORY
# ==================================================

with tab4:
    st.markdown('<div class="section-header">📁 Prediction History & Export</div>', unsafe_allow_html=True)
    
    if "prediction_history" not in st.session_state or len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Run a prediction to start building history.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.markdown(f"### 📊 Total Predictions: {len(history_df)}")
        
        # Display history table
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Export options
        st.markdown("### 💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_to_csv()
            if csv_data:
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            json_data = json.dumps(st.session_state.prediction_history, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.rerun()

# ==================================================
# TAB 5: ABOUT PROJECT
# ==================================================

with tab5:
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 Project Overview", expanded=True):
        st.markdown("""
        ### AI-Powered Diabetes Risk Assessment System
        
        This is a **production-ready healthcare dashboard** that leverages machine learning 
        to predict diabetes risk using clinical health indicators.
        
        **Key Objectives:**
        - Democratize diabetes risk screening
        - Provide data-driven health insights
        - Generate personalized recommendations
        - Enable preventive healthcare decisions
        
        **Target Users:**
        - Individuals concerned about diabetes risk
        - Healthcare providers
        - Wellness programs
        - Research institutions
        """)
    
    with st.expander("📊 Dataset & Model"):
        st.markdown("""
        ### Model Information
        
        **Algorithm:** Random Forest Classifier
        
        **Training Data:** Pima Indians Diabetes Database
        - 768 patient records
        - 8 clinical features
        - Binary classification task
        
        **Model Performance:**
        - High accuracy on test set
        - Robust risk stratification
        - Calibrated probability estimates
        """)
    
    with st.expander("🔧 Features Used"):
        st.markdown("""
        ### Input Features (10 Total)
        
        **Clinical Measurements:**
        1. **Pregnancies** - Number of times pregnant
        2. **Glucose** - Fasting blood glucose (mg/dL)
        3. **Blood Pressure** - Diastolic BP (mmHg)
        4. **Skin Thickness** - Triceps skin fold (mm)
        5. **Insulin** - 2-Hour serum insulin (mIU/mL)
        6. **BMI** - Body Mass Index (kg/m²)
        7. **Diabetes Pedigree Function** - Family history score
        8. **Age** - Age in years
        
        **Derived Features:**
        9. **Age Group** - Categorical age grouping
        10. **BMI Category** - Categorical BMI classification
        """)
    
    with st.expander("💻 Technology Stack"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Backend:**
            - Python 3.8+
            - Scikit-Learn
            - NumPy
            - Pandas
            """)
        
        with col2:
            st.markdown("""
            **Frontend:**
            - Streamlit
            - Matplotlib
            - Seaborn
            - Plotly
            """)
        
        with col3:
            st.markdown("""
            **Deployment:**
            - Streamlit Cloud
            - Docker
            - GitHub
            - CI/CD Ready
            """)
    
    with st.expander("📋 Use Cases"):
        st.markdown("""
        ### Real-World Applications
        
        1. **Telemedicine Platforms** - Initial risk screening before consultations
        2. **Wellness Programs** - Corporate health initiative
        3. **Community Health** - Screening campaigns
        4. **Personal Health Monitoring** - Individual wellness tracking
        5. **Medical Research** - Data collection and analysis
        6. **Education** - Teaching ML applications in healthcare
        """)
    
    with st.expander("⚠️ Important Disclaimer"):
        st.markdown("""
        ### Medical Disclaimer
        
        **This tool is for educational and informational purposes only.**
        
        ❌ **NOT a substitute for professional medical diagnosis**
        ❌ **NOT a replacement for consultation with healthcare providers**
        ❌ **Results should not be used for medical decision-making alone**
        
        ✅ **Always consult qualified healthcare professionals for:**
        - Diagnosis confirmation
        - Treatment plans
        - Medical advice
        - Prescription decisions
        
        **Limitations:**
        - Model trained on specific population (Pima Indians)
        - May not generalize to all populations
        - Accuracy depends on input data quality
        - Should be used with clinical judgment
        """)
    
    st.divider()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
    <p>🩺 <b>AI Diabetes Risk Assessment</b> | Made with ❤️ using Streamlit</p>
    <p>For medical concerns, please consult a healthcare professional.</p>
    </div>
    """, unsafe_allow_html=True)
