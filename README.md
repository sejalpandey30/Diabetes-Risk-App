# 🩺 AI Diabetes Risk Assessment System

An AI-powered healthcare dashboard that predicts diabetes risk using Machine Learning and provides personalized health recommendations through an interactive web interface.

## 🚀 Live Demo

[https://diabetes-risk-app-001.streamlit.app/]

---

## 📌 Project Overview

This project uses clinical health indicators to predict the likelihood of diabetes in patients.

The system combines:

- Data Analysis
- Feature Engineering
- Machine Learning
- Risk Assessment
- Personalized Recommendations
- Interactive Dashboard

The application is designed to assist in early diabetes risk screening and health awareness.

---

## ✨ Features

### 🔍 Diabetes Risk Prediction
Predicts the probability of diabetes using patient health information.

### 📊 Risk Analysis Dashboard
Displays:
- Risk Probability
- Risk Category
- Prediction Confidence

### 💡 Personalized Recommendations
Provides lifestyle recommendations based on:
- Glucose levels
- BMI
- Blood Pressure
- Age

### 📈 Interactive Visualization
Visual representation of prediction confidence and risk assessment.

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing
- Missing value handling
- Data cleaning
- Feature scaling

### Feature Engineering
Additional features generated:
- Age Group
- BMI Category

### Model Training
Machine Learning algorithms evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting

Final model:
- Random Forest Classifier

---

## 📋 Features Used

| Feature | Description |
|----------|------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes likelihood |
| Age | Patient age |
| AgeGroup | Engineered feature |
| BMI_Category | Engineered feature |

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Scikit-Learn
- NumPy
- Pandas

### Visualization
- Matplotlib

### Web Application
- Streamlit

---

## 📂 Project Structure
Diabetes-Risk-App/

├── app.py
├── diabetes_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── feature_importance.


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Diabetes-Risk-App.git

cd Diabetes-Risk-App
````

📸 Application Preview

Dashboard

<img width="904" height="350" alt="image" src="https://github.com/user-attachments/assets/029d40db-c2fa-44e3-929f-0002b31cd80c" />


Prediction Results

<img width="929" height="422" alt="image" src="https://github.com/user-attachments/assets/e526abd4-9efc-4116-9eae-0c127442a8ef" />

Recommendation Page

<img width="811" height="395" alt="image" src="https://github.com/user-attachments/assets/0aedc154-35d2-4083-98ca-56341610e5ef" />

Assessment Analysis

<img width="911" height="338" alt="image" src="https://github.com/user-attachments/assets/f6428d6f-2a00-4582-815b-eeb295c809e5" />

🎯 Future Improvements
SHAP Explainability
PDF Health Reports
Patient History Tracking
Database Integration
Multi-Page Dashboard
Advanced Risk Analytics

👨‍💻 Author

Sejal Pandey


📜 License

This project is developed for educational and portfolio purposes.


---


