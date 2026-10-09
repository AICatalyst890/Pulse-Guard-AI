import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="PulseGuard AI - Health Risk Triage",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for high-visibility UI styling across Light & Dark themes
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    div[data-testid="stBlock"] {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD MACHINE LEARNING MODEL ARTIFACTS
# ==============================================================================
@st.cache_resource
def load_trained_models():
    """Loads saved Scikit-Learn pipelines from the models directory."""
    try:
        heart_model = joblib.load('models/heart_model.pkl')
        diabetes_model = joblib.load('models/diabetes_model.pkl')
        return heart_model, diabetes_model
    except Exception as e:
        st.error(f"⚠️ Error loading model files from 'models/' folder: {e}")
        st.stop()

heart_model, diabetes_model = load_trained_models()

# ==============================================================================
# 3. DYNAMIC RULE-BASED CLINICAL RECOMMENDATION ENGINE
# ==============================================================================
def generate_health_tips(age, sex, heart_prob, diab_prob, trestbps, chol, glucose, bmi, fbs_val):
    """
    Generates tailored, factor-specific clinical action plans based on individual biometrics.
    """
    diet_steps = []
    diagnostic_tests = []
    lifestyle_goals = []

    # --- 1. Blood Pressure (trestbps) Factors ---
    if trestbps >= 140:
        diet_steps.append("**High BP Alert:** Strictly limit sodium intake (< 1,500 mg/day). Avoid processed foods and added salt.")
        diagnostic_tests.append("**Blood Pressure Monitoring:** Record resting BP twice daily for 7 days and consult a physician.")
    elif trestbps >= 120:
        diet_steps.append("**Elevated BP:** Adopt a DASH-style diet rich in potassium (spinach, bananas) and cut down on dietary salt.")
        diagnostic_tests.append("Re-check resting blood pressure in 7 to 10 days.")
    else:
        diet_steps.append("**Normal BP:** Maintain balanced salt and fluid intake.")

    # --- 2. Cholesterol (chol) Factors ---
    if chol >= 240:
        diet_steps.append("**High Cholesterol:** Eliminate trans fats and fried foods. Increase soluble fiber (oats, legumes, flaxseeds).")
        diagnostic_tests.append("Schedule a comprehensive Lipid Profile (HDL/LDL/Triglycerides) blood panel at the PHC.")
    elif chol >= 200:
        diet_steps.append("**Borderline High Cholesterol:** Limit saturated fats, butter, and full-fat dairy cream.")
    else:
        diet_steps.append("**Normal Cholesterol:** Maintain healthy fat balance in meals.")

    # --- 3. Blood Glucose & Fasting Sugar Factors ---
    if glucose >= 140 or fbs_val == 1:
        diet_steps.append("**Elevated Glucose:** Strictly restrict refined sugars, sweets, fruit juices, and high-GI carbohydrates.")
        diagnostic_tests.append("Schedule an HbA1c fasting blood test within 3–5 days to evaluate long-term glycemic control.")
    elif glucose >= 100:
        diet_steps.append("**Borderline Fasting Glucose:** Control carbohydrate portion sizes and balance meals with lean protein and fiber.")
    else:
        diet_steps.append("**Normal Glucose:** Fasting sugar levels are within normal range.")

    # --- 4. BMI Factors ---
    if bmi >= 30:
        lifestyle_goals.append("**Obese BMI Range:** Target a progressive 5–10% body weight reduction over the next 3 to 6 months.")
    elif bmi >= 25:
        lifestyle_goals.append("**Overweight BMI Range:** Focus on portion control and reducing caloric intake from processed snacks.")
    elif bmi < 18.5:
        lifestyle_goals.append("**Underweight BMI Range:** Ensure adequate calorie-dense and protein-rich intake.")
    else:
        lifestyle_goals.append("**Healthy BMI Range:** Continue maintaining balanced caloric intake.")

    # --- 5. Age & Disease Risk Score Factors ---
    if heart_prob >= 60:
        diagnostic_tests.append("🚨 **High Heart Risk (>=60%):** Immediate medical evaluation and 12-lead ECG recommended.")
    elif heart_prob >= 35:
        diagnostic_tests.append("⚠️ **Moderate Heart Risk:** Schedule a routine cardiac evaluation within 2 weeks.")

    if diab_prob >= 60:
        diagnostic_tests.append("🚨 **High Diabetes Risk (>=60%):** Requires prompt clinical consultation for metabolic management.")

    if age >= 50:
        lifestyle_goals.append("**Age 50+ Care:** Incorporate low-impact joint exercises and schedule yearly comprehensive health screenings.")

    # --- 6. Physical Activity Goals ---
    if heart_prob < 60:
        lifestyle_goals.append("Aim for at least 30 minutes of moderate aerobic activity (e.g., brisk walking) 5 days a week.")
    else:
        lifestyle_goals.append("Begin light walking only after obtaining clearance from a doctor.")

    # Format lists into clear Markdown bullet points
    diet_fmt = "\n".join([f"- {item}" for item in diet_steps])
    diag_fmt = "\n".join([f"- {item}" for item in diagnostic_tests])
    life_fmt = "\n".join([f"- {item}" for item in lifestyle_goals])

    return diet_fmt, diag_fmt, life_fmt

# ==============================================================================
# 4. STREAMLIT FRONTEND USER INTERFACE
# ==============================================================================

# App Title & Subtitle
st.markdown('<div class="main-title">🩺 PulseGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Early Chronic Disease Risk Assessment & Health Guidance | SDG 3: Good Health and Well-Being</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Biometric Input Form
st.sidebar.header("📋 Patient Clinical Inputs")
st.sidebar.markdown("Enter biometrics captured during the health screening:")

# Demographics
age = st.sidebar.slider("Age (Years)", 18, 90, 48)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
sex_val = 1 if sex == "Male" else 0

# Cardiovascular Vitals
st.sidebar.subheader("❤️ Cardiovascular Parameters")
cp = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_mapping = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
cp_val = cp_mapping[cp]

trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 210, 130)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dL)", 100, 500, 220)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
fbs_val = 1 if fbs == "Yes" else 0

restecg = st.sidebar.selectbox(
    "Resting ECG Results",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)
restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg_val = restecg_mapping[restecg]

thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang_val = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.number_input("ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.2)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
slope_mapping = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
slope_val = slope_mapping[slope]

ca = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia Status", ["Normal", "Fixed Defect", "Reversible Defect"])
thal_mapping = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
thal_val = thal_mapping[thal]

# Diabetes Vitals
st.sidebar.subheader("🩸 Diabetes & Metabolic Parameters")
glucose = st.sidebar.number_input("Plasma Glucose Concentration (mg/dL)", 60, 300, 125)
bmi = st.sidebar.number_input("Body Mass Index (BMI)", 15.0, 50.0, 28.4)
insulin = st.sidebar.number_input("2-Hour Serum Insulin (mu U/mL)", 10, 400, 85)
dpf = st.sidebar.number_input("Diabetes Pedigree Function Score", 0.05, 2.50, 0.47)
pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 17, 0)
skin_thickness = st.sidebar.number_input("Triceps Skin Fold Thickness (mm)", 10, 99, 20)

# Main Action Button
if st.button("🔍 Run Dual Risk Assessment & Get Recommendations", type="primary", use_container_width=True):

    # Construct DataFrames matching model feature columns
    heart_input = pd.DataFrame([[
        age, sex_val, cp_val, trestbps, chol, fbs_val,
        restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val
    ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    diab_input = pd.DataFrame([[
        pregnancies, glucose, trestbps, skin_thickness, insulin, bmi, dpf, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Calculate Risk Scores
    heart_prob = heart_model.predict_proba(heart_input)[0][1] * 100
    diab_prob = diabetes_model.predict_proba(diab_input)[0][1] * 100

    # Display Analytics Dashboard
    st.subheader("📊 Dual Disease Risk Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ❤️ Cardiovascular Risk Score")
        st.metric("Heart Risk Score", f"{heart_prob:.1f}%")
        st.progress(int(heart_prob))
        if heart_prob >= 60:
            st.error("🚨 High Risk: Immediate specialist evaluation advised")
        elif heart_prob >= 35:
            st.warning("⚠️ Moderate Risk: Regular monitoring required")
        else:
            st.success("✅ Low Risk: Normal Cardiovascular Profile")

    with col2:
        st.markdown("### 🩸 Diabetes Risk Score")
        st.metric("Diabetes Risk Score", f"{diab_prob:.1f}%")
        st.progress(int(diab_prob))
        if diab_prob >= 60:
            st.error("🚨 High Risk: Immediate glucose management needed")
        elif diab_prob >= 35:
            st.warning("⚠️ Moderate Risk: Dietary adjustments recommended")
        else:
            st.success("✅ Low Risk: Normal Metabolic Profile")

    st.markdown("---")

    # Display Recommendations & Health Guidance
    st.subheader("💡 Personalized Clinical Health Guidance & Action Plan")

    # Patient Summary Card via Streamlit Container (Theme Independent)
    with st.container(border=True):
        st.markdown(f"""
        ### 📋 Patient Assessment Summary
        - **Demographics:** {age} years old | **Gender:** {sex}
        - **Model Predictions:** Heart Disease Risk: **{heart_prob:.1f}%** | Diabetes Risk: **{diab_prob:.1f}%**
        - **Key Vitals:** Blood Pressure: **{trestbps} mm Hg** | Fasting Glucose: **{glucose} mg/dL** | Cholesterol: **{chol} mg/dL** | BMI: **{bmi}**
        """)

    # Generate Dynamic Rule-Based Tips
    diet_fmt, diag_fmt, life_fmt = generate_health_tips(
        age, sex, heart_prob, diab_prob, trestbps, chol, glucose, bmi, fbs_val
    )

    # Output Structured Sections inside Clean Native Expanders/Containers
    st.markdown("#### 🍏 1. Dietary & Nutrition Plan")
    st.markdown(diet_fmt)

    st.markdown("#### 🔬 2. Recommended Clinical Next Steps")
    st.markdown(diag_fmt)

    st.markdown("#### 🏃 3. Lifestyle & Actionable Goals")
    st.markdown(life_fmt)

else:
    st.info("👈 Adjust patient parameters on the left sidebar and click **'Run Dual Risk Assessment & Get Recommendations'** to calculate risk scores.")