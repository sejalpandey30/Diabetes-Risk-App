import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Diabetes Risk Assessment - Healthcare Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL CSS STYLING
# ==================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        padding-top: 2rem;
        background-color: #f9fafb;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1f2937;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.875rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.375rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    p {
        color: #4b5563;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Section header */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        color: #0f172a;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #f7fee7 100%);
        border-left: 4px solid #22c55e;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid #ef4444;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #d1d5db;
    }
    
    /* Risk indicators */
    .risk-low {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 2px solid #10b981;
        padding: 1.5rem;
        border-radius: 12px;
        color: #065f46;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 12px;
        color: #92400e;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        color: #7f1d1d;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        font-family: 'Inter', sans-serif;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px 8px 0 0;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Data frames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .sidebar .sidebar-content h1 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Text styling */
    .caption {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 400;
    }
    
    /* Divider */
    .divider {
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 2rem 0;
        font-size: 0.9rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
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
        return "Low Risk", "low"
    elif probability < 0.70:
        return "Moderate Risk", "moderate"
    else:
        return "High Risk", "high"

def get_health_recommendations(glucose, bmi, age, bp, insulin, dpf):
    """Generate personalized health recommendations"""
    recommendations = []
    risk_factors = []
    
    if glucose > 140:
        risk_factors.append("Elevated Glucose Level")
        recommendations.append({
            "category": "Nutrition",
            "advice": "Reduce refined sugars and carbohydrates. Increase fiber intake through whole grains and vegetables.",
            "priority": "HIGH"
        })
    
    if bmi > 25:
        risk_factors.append("Overweight or Obesity")
        recommendations.append({
            "category": "Physical Activity",
            "advice": f"Increase physical activity to at least 150 minutes per week. Current BMI: {bmi:.1f} (Target: 18.5-24.9)",
            "priority": "HIGH"
        })
    
    if age > 45:
        risk_factors.append("Age Group (45+)")
        recommendations.append({
            "category": "Medical Screening",
            "advice": "Schedule annual diabetes screening and comprehensive health check-ups.",
            "priority": "HIGH"
        })
    
    if bp > 130:
        risk_factors.append("Elevated Blood Pressure")
        recommendations.append({
            "category": "Cardiovascular Health",
            "advice": "Monitor blood pressure regularly. Reduce sodium intake and manage stress through meditation or exercise.",
            "priority": "HIGH"
        })
    
    if insulin > 250:
        risk_factors.append("High Insulin Levels")
        recommendations.append({
            "category": "Metabolic Health",
            "advice": "Consult with an endocrinologist. May indicate insulin resistance requiring professional evaluation.",
            "priority": "HIGH"
        })
    
    if dpf > 1.0:
        risk_factors.append("Strong Family History")
    
    if not risk_factors:
        recommendations.append({
            "category": "General Wellness",
            "advice": "Continue maintaining your current healthy lifestyle and regular health check-ups.",
            "priority": "LOW"
        })
    
    return recommendations, risk_factors

def save_prediction_to_history(prediction_data):
    """Save prediction to session history"""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_data["timestamp"] = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    st.session_state.prediction_history.append(prediction_data)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## Diabetes Risk Assessment")
    st.markdown("---")
    
    st.markdown("""
    ### About This Tool
    
    A clinical-grade diabetes risk prediction system powered by machine learning and evidence-based medical guidelines.
    """)
    
    st.markdown("---")
    
    st.markdown("### Key Features")
    st.markdown("• Risk Prediction & Analysis")
    st.markdown("• Personalized Recommendations")
    st.markdown("• Trend Tracking & Analytics")
    st.markdown("• Data Export Capabilities")
    
    st.markdown("---")
    
    st.markdown("### Technology")
    st.markdown("""
    - **Algorithm:** Random Forest Classifier
    - **Framework:** Streamlit
    - **ML Library:** Scikit-Learn
    - **Data Processing:** Pandas, NumPy
    """)
    
    if "prediction_history" in st.session_state and len(st.session_state.prediction_history) > 0:
        st.markdown("---")
        st.markdown("### Session Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Assessments", len(st.session_state.prediction_history))
        with col2:
            latest_risk = st.session_state.prediction_history[-1]['risk_probability'] * 100
            st.metric("Latest Risk", f"{latest_risk:.1f}%")

# ==================================================
# MAIN HEADER
# ==================================================

st.markdown("# Diabetes Risk Assessment")
st.markdown("#### Clinical-Grade Prediction Using Machine Learning")

st.markdown("""
<div class="info-box">
This assessment tool provides a preliminary diabetes risk prediction based on clinical health metrics. 
Results are for informational purposes and should not replace professional medical advice.
</div>
""", unsafe_allow_html=True)

# ==================================================
# TOP METRICS
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Algorithm", "Random Forest", delta=None)

with col2:
    st.metric("Input Features", "10 Variables", delta=None)

with col3:
    st.metric("Classification", "Binary", delta=None)

with col4:
    total_pred = len(st.session_state.get("prediction_history", []))
    st.metric("Session Count", str(total_pred), delta=None)

st.markdown("---")

# ==================================================
# MAIN TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Assessment",
    "Recommendations",
    "Analytics",
    "History",
    "Information"
])

# ==================================================
# TAB 1: ASSESSMENT
# ==================================================

with tab1:
    st.markdown("## Health Assessment Form")
    
    st.markdown("""
    <div class="info-box">
    Enter your health metrics for a comprehensive diabetes risk assessment. All fields are required.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Clinical Measurements")
        
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Total pregnancies (applicable for biological females)"
        )
        
        glucose = st.number_input(
            "Fasting Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
            help="Normal: <100 | Prediabetic: 100-125 | Diabetic: ≥126"
        )
        
        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=0,
            max_value=200,
            value=70,
            help="Normal: <80 | Elevated: 80-89 | High: ≥90"
        )
        
        skin_thickness = st.number_input(
            "Triceps Skin Fold (mm)",
            min_value=0,
            max_value=100,
            value=20,
            help="Measure of subcutaneous fat"
        )
    
    with col2:
        st.markdown("### Additional Metrics")
        
        insulin = st.number_input(
            "2-Hour Insulin (mIU/mL)",
            min_value=0,
            max_value=900,
            value=80,
            help="Normal: <12 | Elevated: 12-150 | High: >150"
        )
        
        bmi = st.number_input(
            "Body Mass Index (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Underweight: <18.5 | Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: ≥30"
        )
        
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Family history score (0-5 scale)"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=100,
            value=30,
            help="Your current age"
        )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        predict_button = st.button(
            "Generate Risk Assessment",
            use_container_width=True,
            type="primary"
        )
    
    with col_btn2:
        if st.button("Clear Form", use_container_width=True):
            st.session_state.pop("probability", None)
            st.session_state.pop("prediction_data", None)
            st.rerun()
    
    # Prediction Logic
    if predict_button:
        age_group = get_age_group(age)
        bmi_category = get_bmi_category(bmi)
        
        patient_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, age_group, bmi_category
        ]])
        
        try:
            patient_scaled = scaler.transform(patient_data)
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0][1]
            
            st.session_state["probability"] = probability
            st.session_state["prediction_data"] = {
                "pregnancies": pregnancies, "glucose": glucose,
                "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
                "insulin": insulin, "bmi": bmi, "dpf": dpf, "age": age,
                "risk_probability": probability
            }
            
            save_prediction_to_history(st.session_state["prediction_data"])
            
        except Exception as e:
            st.error(f"Assessment Error: {str(e)}")
    
    # Display Results
    if "probability" in st.session_state:
        st.markdown("---")
        st.markdown("## Assessment Results")
        
        probability = st.session_state["probability"]
        risk_label, risk_level = get_risk_level(probability)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Probability", f"{probability*100:.2f}%", delta=None)
        
        with col2:
            confidence = (1 - abs(probability - 0.5) * 2) * 100
            st.metric("Confidence Level", f"{confidence:.1f}%", delta=None)
        
        with col3:
            st.metric("Risk Category", risk_label, delta=None)
        
        st.markdown("---")
        
        # Risk Display
        st.markdown(f'<div class="risk-{risk_level}"><strong>{risk_label.upper()}</strong></div>', unsafe_allow_html=True)
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("### Risk Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#f9fafb')
            ax.set_facecolor('#ffffff')
            
            categories = ["Negative", "Positive"]
            values = [(1-probability)*100, probability*100]
            colors = ["#10b981", "#ef4444"]
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            ax.set_ylabel("Probability (%)", fontweight="600", fontsize=11)
            ax.set_title("Prediction Distribution", fontweight="700", fontsize=12, pad=20)
            ax.set_ylim(0, 100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#e5e7eb')
            ax.spines['bottom'].set_color('#e5e7eb')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontweight='600', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_viz2:
            st.markdown("### Health Metrics Status")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#f9fafb')
            ax.set_facecolor('#ffffff')
            
            pred = st.session_state["prediction_data"]
            metrics = ["Glucose", "BMI", "BP", "Insulin", "Age"]
            normalized = [
                min(pred['glucose']/300 * 100, 100),
                min(pred['bmi']/30 * 100, 100),
                min(pred['blood_pressure']/200 * 100, 100),
                min(pred['insulin']/900 * 100, 100),
                min(pred['age']/100 * 100, 100)
            ]
            
            bars = ax.barh(metrics, normalized, color="#3b82f6", alpha=0.8, edgecolor='white', linewidth=2)
            ax.set_xlabel("Relative Level (%)", fontweight="600", fontsize=11)
            ax.set_title("Normalized Metrics", fontweight="700", fontsize=12, pad=20)
            ax.set_xlim(0, 100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#e5e7eb')
            ax.spines['bottom'].set_color('#e5e7eb')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.0f}%', ha='left', va='center', fontweight='600', fontsize=9, color='#1f2937')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================================================
# TAB 2: RECOMMENDATIONS
# ==================================================

with tab2:
    st.markdown("## Personalized Health Recommendations")
    
    if "prediction_data" not in st.session_state:
        st.markdown("""
        <div class="info-box">
        Complete an assessment to receive personalized health recommendations.
        </div>
        """, unsafe_allow_html=True)
    else:
        pred_data = st.session_state["prediction_data"]
        recommendations, risk_factors = get_health_recommendations(
            pred_data["glucose"], pred_data["bmi"], pred_data["age"],
            pred_data["blood_pressure"], pred_data["insulin"], pred_data["dpf"]
        )
        
        if risk_factors:
            st.markdown("### Identified Health Factors")
            cols = st.columns(min(len(risk_factors), 3))
            for idx, factor in enumerate(risk_factors[:3]):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="warning-box">
                    <strong>{factor}</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Clinical Recommendations")
        
        for rec in recommendations:
            with st.expander(f"**{rec['category']}** - {rec['priority']}", expanded=(rec['priority']=='HIGH')):
                st.markdown(f"**Recommendation:** {rec['advice']}")
        
        st.markdown("---")
        
        st.markdown("### General Health Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Dietary Modifications
            - Emphasize whole grains over refined carbohydrates
            - Include lean proteins in every meal
            - Increase vegetable and fiber intake
            - Limit added sugars and processed foods
            - Maintain consistent meal timing
            
            #### Physical Activity
            - Aim for 150 minutes of moderate aerobic activity weekly
            - Include resistance training 2-3 times per week
            - Reduce sedentary time throughout the day
            - Incorporate daily walking (10,000 steps target)
            """)
        
        with col2:
            st.markdown("""
            #### Lifestyle Factors
            - Maintain 7-9 hours of quality sleep nightly
            - Practice stress management techniques
            - Limit alcohol consumption
            - Avoid tobacco products
            - Monitor weight regularly
            
            #### Medical Monitoring
            - Schedule annual health check-ups
            - Regular blood glucose monitoring if recommended
            - Blood pressure tracking
            - Annual lipid panel screening
            """)

# ==================================================
# TAB 3: ANALYTICS
# ==================================================

with tab3:
    st.markdown("## Assessment Analytics")
    
    if "prediction_data" not in st.session_state:
        st.info("Complete an assessment to view analytics.")
    else:
        pred_data = st.session_state["prediction_data"]
        probability = st.session_state["probability"]
        
        st.markdown("### Current Health Metrics Overview")
        metrics_df = pd.DataFrame({
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
                "✓ Normal" if pred_data['glucose'] < 100 else "⚠ Elevated" if pred_data['glucose'] < 126 else "✗ High",
                "✓ Normal" if pred_data['bmi'] < 25 else "⚠ Overweight" if pred_data['bmi'] < 30 else "✗ Obese",
                "✓ Normal" if pred_data['blood_pressure'] < 120 else "⚠ Elevated" if pred_data['blood_pressure'] < 140 else "✗ High",
                "✓ Normal" if pred_data['insulin'] < 12 else "⚠ Elevated" if pred_data['insulin'] < 150 else "✗ High",
                "✓ <45" if pred_data['age'] < 45 else "⚠ 45-60" if pred_data['age'] < 60 else "✗ >60",
                "✓ Low" if pred_data['dpf'] < 0.5 else "⚠ Medium" if pred_data['dpf'] < 1.0 else "✗ High"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        probability_val = st.session_state["probability"]
        if probability_val < 0.30:
            st.markdown("""
            <div class="success-box">
            <h3>✓ Low Risk Profile</h3>
            Continue with current healthy lifestyle habits. Maintain annual health screenings.
            </div>
            """, unsafe_allow_html=True)
        elif probability_val < 0.70:
            st.markdown("""
            <div class="warning-box">
            <h3>⚠ Moderate Risk Profile</h3>
            Implement lifestyle modifications targeting identified risk factors. Schedule follow-up with healthcare provider.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
            <h3>✗ High Risk Profile</h3>
            Recommend consultation with healthcare provider. Consider comprehensive diabetes screening and aggressive lifestyle intervention.
            </div>
            """, unsafe_allow_html=True)
        
        if len(st.session_state.get("prediction_history", [])) > 1:
            st.markdown("---")
            st.markdown("### Risk Assessment Trend")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#f9fafb')
            ax.set_facecolor('#ffffff')
            
            ax.plot(range(len(history_df)), history_df['risk_probability'] * 100,
                   marker='o', linewidth=2.5, markersize=8, color='#3b82f6', label='Risk Trend')
            ax.fill_between(range(len(history_df)), history_df['risk_probability'] * 100, alpha=0.15, color='#3b82f6')
            ax.set_ylabel("Risk Probability (%)", fontweight="600", fontsize=11)
            ax.set_xlabel("Assessment Number", fontweight="600", fontsize=11)
            ax.set_title("Historical Risk Trend", fontweight="700", fontsize=12, pad=20)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================================================
# TAB 4: HISTORY
# ==================================================

with tab4:
    st.markdown("## Assessment History")
    
    if "prediction_history" not in st.session_state or len(st.session_state.prediction_history) == 0:
        st.info("No assessments yet. Complete an assessment to build your history.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.markdown(f"### Total Assessments: {len(history_df)}")
        
        st.dataframe(
            history_df[['timestamp', 'glucose', 'bmi', 'blood_pressure', 'age', 'risk_probability']],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"assessment_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            import json
            json_data = json.dumps(st.session_state.prediction_history, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"assessment_history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            if st.button("Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.success("History cleared successfully.")
                st.rerun()

# ==================================================
# TAB 5: INFORMATION
# ==================================================

with tab5:
    st.markdown("## About This Assessment Tool")
    
    with st.expander("Project Overview", expanded=True):
        st.markdown("""
        ### Clinical Diabetes Risk Assessment System
        
        This tool provides a preliminary evaluation of diabetes risk using machine learning 
        algorithms trained on clinically validated health data.
        
        **Purpose:**
        - Early identification of diabetes risk factors
        - Promotion of preventive healthcare
        - Evidence-based health guidance
        - Support for informed medical decisions
        """)
    
    with st.expander("Methodology & Model"):
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        
        **Dataset:** Pima Indians Diabetes Database
        - 768 clinical records
        - Evidence-based health indicators
        - Binary classification (Diabetes/No Diabetes)
        
        **Model Characteristics:**
        - Trained on validated clinical data
        - Optimized for sensitivity and specificity
        - Calibrated probability estimates
        """)
    
    with st.expander("Input Variables"):
        st.markdown("""
        **Clinical Measurements (8 Primary):**
        1. Number of Pregnancies
        2. Fasting Glucose Level
        3. Diastolic Blood Pressure
        4. Triceps Skin Fold Thickness
        5. 2-Hour Serum Insulin
        6. Body Mass Index
        7. Diabetes Pedigree Function
        8. Age
        
        **Derived Variables (2 Secondary):**
        9. Age Group Classification
        10. BMI Category Classification
        """)
    
    with st.expander("Technical Stack"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Core Technologies**
            - Python 3.8+
            - Streamlit Framework
            - Scikit-Learn ML
            - NumPy & Pandas
            """)
        
        with col2:
            st.markdown("""
            **Visualization**
            - Matplotlib
            - Data Tables
            - Statistical Charts
            - Trend Analysis
            """)
        
        with col3:
            st.markdown("""
            **Deployment**
            - Streamlit Cloud
            - GitHub Integration
            - CI/CD Pipeline
            - Responsive Design
            """)
    
    with st.expander("Medical Disclaimer"):
        st.markdown("""
        ### Important Legal Notice
        
        **⚠️ This assessment tool is for educational purposes only.**
        
        **NOT:**
        - A medical diagnosis
        - A substitute for professional medical evaluation
        - A replacement for healthcare provider consultation
        - Basis for medical treatment decisions
        
        **ALWAYS:**
        - Consult qualified healthcare professionals
        - Seek medical advice for health concerns
        - Use clinical judgment in medical decisions
        - Follow professional medical recommendations
        
        **Limitations:**
        - Model trained on specific population
        - Results depend on data accuracy
        - Individual variations not captured
        - Should be used with professional guidance
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
    <strong>Diabetes Risk Assessment</strong> | Clinical-Grade Prediction System
    <br>
    For medical concerns, please consult a qualified healthcare professional.
    <br>
    © 2024 | All Rights Reserved
    </div>
    """, unsafe_allow_html=True)
