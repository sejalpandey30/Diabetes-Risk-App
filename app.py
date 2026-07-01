import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        border: none;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model_and_scaler():
    """Load ML model and scaler"""
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found.")
        st.stop()

model, scaler = load_model_and_scaler()

# ==================================================
# UTILITY FUNCTIONS
# ==================================================

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
    """Determine risk level"""
    if probability < 0.30:
        return "🟢 LOW RISK", "low"
    elif probability < 0.70:
        return "🟡 MODERATE RISK", "moderate"
    else:
        return "🔴 HIGH RISK", "high"

def get_health_recommendations(glucose, bmi, age, bp, insulin, dpf):
    """Generate personalized health recommendations"""
    recommendations = []
    risk_factors = []
    
    if glucose > 140:
        risk_factors.append("High Glucose")
        recommendations.append({
            "category": "🍎 Nutrition",
            "advice": "Reduce sugar and refined carbohydrate intake. Focus on whole grains.",
            "priority": "HIGH"
        })
    
    if bmi > 25:
        risk_factors.append("Overweight/Obesity")
        recommendations.append({
            "category": "💪 Exercise",
            "advice": f"Increase physical activity to 150 min/week. Current BMI: {bmi:.1f}",
            "priority": "HIGH"
        })
    
    if age > 45:
        risk_factors.append("Age > 45")
        recommendations.append({
            "category": "🏥 Screening",
            "advice": "Schedule annual diabetes screening and health check-ups.",
            "priority": "HIGH"
        })
    
    if bp > 130:
        risk_factors.append("High Blood Pressure")
        recommendations.append({
            "category": "❤️ Cardiovascular",
            "advice": "Monitor blood pressure. Reduce sodium intake.",
            "priority": "HIGH"
        })
    
    if insulin > 250:
        risk_factors.append("High Insulin")
        recommendations.append({
            "category": "🧬 Metabolic",
            "advice": "Consult endocrinologist. May indicate insulin resistance.",
            "priority": "HIGH"
        })
    
    if dpf > 1.0:
        risk_factors.append("Strong Family History")
    
    if not risk_factors:
        recommendations.append({
            "category": "✅ General",
            "advice": "Maintain current healthy lifestyle.",
            "priority": "LOW"
        })
    
    return recommendations, risk_factors

def save_prediction_to_history(prediction_data):
    """Save prediction to session history"""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_history.append(prediction_data)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.title("🩺 Diabetes AI")
    st.markdown("---")
    
    st.info("""
    AI-powered diabetes risk prediction system.
    
    **Built with:**
    - Machine Learning
    - Streamlit
    - Scikit-Learn
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Features")
    st.write("✅ Risk Prediction")
    st.write("✅ Recommendations")
    st.write("✅ Analytics")
    st.write("✅ History Tracking")
    
    st.markdown("---")
    
    if "prediction_history" in st.session_state and len(st.session_state.prediction_history) > 0:
        st.subheader("📈 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predictions", len(st.session_state.prediction_history))
        with col2:
            latest_risk = st.session_state.prediction_history[-1]['risk_probability'] * 100
            st.metric("Latest Risk", f"{latest_risk:.1f}%")

# ==================================================
# MAIN HEADER
# ==================================================

st.markdown('<div class="section-header">🩺 AI Diabetes Risk Assessment Dashboard</div>', unsafe_allow_html=True)

st.markdown(
    "💡 Predict diabetes risk using machine learning and receive personalized health insights."
)

# ==================================================
# TOP METRICS
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🤖 Model", "Random Forest")

with col2:
    st.metric("📊 Features", "10")

with col3:
    st.metric("🎯 Type", "Binary Classification")

with col4:
    total_pred = len(st.session_state.get("prediction_history", []))
    st.metric("📈 Total Predictions", total_pred)

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
    📝 Enter your health metrics for diabetes risk prediction.
    </div>
    """, unsafe_allow_html=True)
    
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
            help="Fasting blood glucose level"
        )
        
        blood_pressure = st.number_input(
            "💓 Blood Pressure (mmHg)",
            min_value=0,
            max_value=200,
            value=70,
            help="Diastolic blood pressure"
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
            help="2-Hour serum insulin level"
        )
        
        bmi = st.number_input(
            "⚖️ BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index"
        )
        
        dpf = st.number_input(
            "👨‍👩‍👧‍👦 Diabetes Pedigree Function",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Family history score"
        )
        
        age = st.number_input(
            "🎂 Age (years)",
            min_value=1,
            max_value=100,
            value=30,
            help="Your age"
        )
    
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        predict_button = st.button(
            "🔍 Assess Diabetes Risk",
            use_container_width=True
        )
    
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.pop("probability", None)
            st.session_state.pop("prediction_data", None)
            st.rerun()
    
    with col_btn3:
        st.write("")
    
    # Prediction Logic
    if predict_button:
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
        
        try:
            patient_scaled = scaler.transform(patient_data)
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0][1]
            
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
            
            save_prediction_to_history(st.session_state["prediction_data"])
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # Display Results
    if "probability" in st.session_state:
        st.divider()
        st.markdown('<div class="section-header">📊 Assessment Results</div>', unsafe_allow_html=True)
        
        probability = st.session_state["probability"]
        risk_label, risk_level = get_risk_level(probability)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Probability", f"{probability*100:.2f}%")
        
        with col2:
            confidence = (1 - abs(probability - 0.5) * 2) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col3:
            risk_display = "🟢 LOW" if risk_level == "low" else "🟡 MODERATE" if risk_level == "moderate" else "🔴 HIGH"
            st.metric("Risk Level", risk_display)
        
        st.divider()
        
        # Risk Display
        st.markdown(f'<div class="risk-{risk_level}"><b>{risk_label}</b></div>', unsafe_allow_html=True)
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("### Risk Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            categories = ["No Diabetes", "Diabetes"]
            values = [(1-probability)*100, probability*100]
            colors = ["#28a745", "#dc3545"]
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
            ax.set_ylabel("Probability (%)", fontweight="bold")
            ax.set_title("Prediction Distribution", fontweight="bold")
            ax.set_ylim(0, 100)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_viz2:
            st.markdown("### Health Metrics")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            pred = st.session_state["prediction_data"]
            metrics = ["Glucose", "BMI", "BP", "Insulin", "Age"]
            normalized = [
                min(pred['glucose']/300 * 100, 100),
                min(pred['bmi']/30 * 100, 100),
                min(pred['blood_pressure']/200 * 100, 100),
                min(pred['insulin']/900 * 100, 100),
                min(pred['age']/100 * 100, 100)
            ]
            
            bars = ax.barh(metrics, normalized, color="#667eea", alpha=0.7)
            ax.set_xlabel("Relative Level (%)", fontweight="bold")
            ax.set_title("Normalized Metrics", fontweight="bold")
            ax.set_xlim(0, 100)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.0f}%', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================================================
# TAB 2: RECOMMENDATIONS
# ==================================================

with tab2:
    st.markdown('<div class="section-header">💊 Health Recommendations</div>', unsafe_allow_html=True)
    
    if "prediction_data" not in st.session_state:
        st.markdown("""
        <div class="info-box">
        ℹ️ Run a prediction first to get personalized recommendations.
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
        
        if risk_factors:
            st.markdown("### ⚠️ Risk Factors")
            for factor in risk_factors:
                st.warning(f"🚨 {factor}")
        
        st.divider()
        
        st.markdown("### 📋 Recommendations")
        
        for rec in recommendations:
            with st.expander(f"{rec['category']} - {rec['priority']}"):
                st.markdown(rec['advice'])
        
        st.divider()
        
        st.markdown("### 📖 General Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🥗 Nutrition:**
            - Eat whole grains
            - Limit sugar
            - Portion control
            
            **💪 Exercise:**
            - 150 min/week
            - Strength training
            - Reduce sitting time
            """)
        
        with col2:
            st.markdown("""
            **😴 Lifestyle:**
            - 7-9 hours sleep
            - Manage stress
            - Limit alcohol
            - No smoking
            
            **🏥 Monitoring:**
            - Annual check-ups
            - BP monitoring
            - Weight tracking
            """)

# ==================================================
# TAB 3: ANALYTICS
# ==================================================

with tab3:
    st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)
    
    if "prediction_data" not in st.session_state:
        st.info("Run a prediction first.")
    else:
        pred_data = st.session_state["prediction_data"]
        probability = st.session_state["probability"]
        
        st.markdown("### Your Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Glucose", "BMI", "Blood Pressure", "Insulin", "Age", "Family History"],
            "Value": [
                f"{pred_data['glucose']}",
                f"{pred_data['bmi']:.1f}",
                f"{pred_data['blood_pressure']}",
                f"{pred_data['insulin']}",
                f"{pred_data['age']}",
                f"{pred_data['dpf']:.2f}"
            ],
            "Status": [
                "🟢" if pred_data['glucose'] < 100 else "🟡" if pred_data['glucose'] < 126 else "🔴",
                "🟢" if pred_data['bmi'] < 25 else "🟡" if pred_data['bmi'] < 30 else "🔴",
                "🟢" if pred_data['blood_pressure'] < 120 else "🟡" if pred_data['blood_pressure'] < 140 else "🔴",
                "🟢" if pred_data['insulin'] < 12 else "🟡" if pred_data['insulin'] < 150 else "🔴",
                "🟢" if pred_data['age'] < 45 else "🟡" if pred_data['age'] < 60 else "🔴",
                "🟢" if pred_data['dpf'] < 0.5 else "🟡" if pred_data['dpf'] < 1.0 else "🔴"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        if probability < 0.30:
            st.success("""
            ### 🟢 LOW RISK
            - Maintain current lifestyle
            - Regular check-ups
            - Continue exercise
            """)
        elif probability < 0.70:
            st.warning("""
            ### 🟡 MODERATE RISK
            - Implement lifestyle changes
            - More frequent monitoring
            - Consult healthcare provider
            """)
        else:
            st.error("""
            ### 🔴 HIGH RISK
            - Consult healthcare provider urgently
            - Consider diabetes screening
            - Aggressive lifestyle changes needed
            """)
        
        if len(st.session_state.get("prediction_history", [])) > 1:
            st.divider()
            st.markdown("### 📈 Trend Over Time")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(history_df)), history_df['risk_probability'] * 100, 
                   marker='o', linewidth=2, markersize=8, color='#667eea')
            ax.fill_between(range(len(history_df)), history_df['risk_probability'] * 100, alpha=0.3)
            ax.set_ylabel("Risk (%)", fontweight="bold")
            ax.set_xlabel("Prediction #", fontweight="bold")
            ax.set_title("Your Risk Trend", fontweight="bold")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================================================
# TAB 4: HISTORY
# ==================================================

with tab4:
    st.markdown('<div class="section-header">📁 Prediction History</div>', unsafe_allow_html=True)
    
    if "prediction_history" not in st.session_state or len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.markdown(f"### Total Predictions: {len(history_df)}")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            import json
            json_data = json.dumps(st.session_state.prediction_history, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.success("History cleared!")
                st.rerun()

# ==================================================
# TAB 5: ABOUT
# ==================================================

with tab5:
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 Project Overview", expanded=True):
        st.markdown("""
        ### AI-Powered Diabetes Risk Assessment
        
        This dashboard uses machine learning to predict diabetes risk based on clinical health indicators.
        
        **Key Features:**
        - Accurate risk prediction
        - Personalized recommendations
        - Health metrics analysis
        - Prediction history tracking
        - Data export capability
        """)
    
    with st.expander("📊 Model & Data"):
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        
        **Training Data:** Pima Indians Diabetes Database
        - 768 patient records
        - 8 clinical features
        - Binary classification
        """)
    
    with st.expander("🔧 Features"):
        st.markdown("""
        **Input Variables (10 total):**
        1. Pregnancies
        2. Glucose
        3. Blood Pressure
        4. Skin Thickness
        5. Insulin
        6. BMI
        7. Diabetes Pedigree Function
        8. Age
        9. Age Group
        10. BMI Category
        """)
    
    with st.expander("💻 Technology"):
        st.markdown("""
        **Stack:**
        - Python 3.8+
        - Streamlit
        - Scikit-Learn
        - NumPy & Pandas
        - Matplotlib
        
        **Deployment:**
        - Streamlit Cloud
        - Docker Ready
        - GitHub Integrated
        """)
    
    with st.expander("⚠️ Disclaimer"):
        st.markdown("""
        ### Important Medical Disclaimer
        
        This tool is for educational purposes only.
        
        ❌ **NOT a medical diagnosis**
        ❌ **NOT a substitute for healthcare providers**
        ❌ **Do not use alone for medical decisions**
        
        ✅ **Always consult qualified healthcare professionals**
        
        **Limitations:**
        - Model trained on specific population
        - Accuracy depends on data quality
        - Should use clinical judgment
        """)
    
    st.divider()
    st.markdown("""
    ---
    <div style="text-align: center;">
    <p>🩺 <b>AI Diabetes Risk Assessment</b> | Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
