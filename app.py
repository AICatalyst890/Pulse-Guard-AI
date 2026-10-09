import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PulseGuard AI | Clinical Health Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# MEDICAL THEME + SIDEBAR VISIBILITY FIX
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F3F7FB;
    color: #172B4D;
}

[data-testid="stHeader"] {
    background-color: #F3F7FB;
}

[data-testid="stMain"] {
    background-color: #F3F7FB;
}

h1, h2, h3, h4 {
    color: #102A43 !important;
}

p {
    color: #334E68;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #102A43 !important;
    border-right: 1px solid #294861;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #102A43 !important;
}

/* Sidebar headings and descriptive text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #FFFFFF !important;
}

/* Critical fix: widget labels such as Age, BP, BMI, Glucose */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Number and text input fields */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
    caret-color: #172B4D !important;
    border: 1px solid #B8C7D9 !important;
    border-radius: 8px !important;
}

/* Input containers */
[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}

/* Dropdown containers */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #B8C7D9 !important;
    border-radius: 8px !important;
}

/* Selected dropdown values */
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
}

/* Dropdown arrow */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #172B4D !important;
    fill: #172B4D !important;
}

/* Dropdown menu options */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="option"] {
    color: #172B4D !important;
    background-color: #FFFFFF !important;
}

/* Radio buttons */
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] p {
    color: #FFFFFF !important;
}

/* Checkboxes */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
    color: #FFFFFF !important;
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr {
    border-color: #456178 !important;
}

/* Main action buttons */
.stButton > button {
    background-color: #0D9488 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
}

/* Cards */
.pg-card {
    background-color: #FFFFFF;
    border: 1px solid #D9E4EF;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(16, 42, 67, 0.05);
}

.pg-card-title {
    color: #102A43;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.pg-muted {
    color: #52677D;
    font-size: 14px;
}

.pg-risk {
    font-size: 28px;
    font-weight: 800;
    margin: 8px 0;
}

.pg-low {
    color: #16803D;
}

.pg-moderate {
    color: #B7791F;
}

.pg-high {
    color: #C53030;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #D9E4EF;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #52677D !important;
}

[data-testid="stMetricValue"] {
    color: #102A43 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
.pg-footer {
    color: #627D98;
    font-size: 12px;
    text-align: center;
    padding: 20px 0 8px 0;
}

/* ===== FIX SELECTED DROPDOWN TEXT ===== */

/* Dropdown Container Background */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #B8C7D9 !important;
    border-radius: 8px !important;
}

/* Selected Text inside Dropdown (Male, Female, Yes, No, etc.) */
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] *,
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Selected option placeholder & value fix */
[data-baseweb="select"] [aria-selected="true"],
[data-baseweb="select"] [aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Dropdown Arrow Symbol */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #000000 !important;
    fill: #000000 !important;
}

/* Open Menu / Options List Box */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] * {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

HEART_MODEL_PATH = MODEL_DIR / "heart_model.pkl"
DIABETES_MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"


@st.cache_resource
def load_model(model_path):
    """Load a saved ML model."""
    return joblib.load(model_path)


def get_model(model_path, display_name):
    """Load a model and display a useful error if unavailable."""
    if not model_path.exists():
        st.error(
            f"{display_name} model not found at: "
            f"`{model_path.relative_to(BASE_DIR)}`"
        )
        return None

    try:
        return load_model(str(model_path))
    except Exception as exc:
        st.error(f"Could not load {display_name} model: {exc}")
        return None


heart_model = get_model(HEART_MODEL_PATH, "Heart disease")
diabetes_model = get_model(DIABETES_MODEL_PATH, "Diabetes")


# =========================================================
# PREDICTION HELPERS
# =========================================================

def get_positive_class_probability(model, input_df):
    """
    Return the probability for the positive class (usually class 1).
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "This model does not support predict_proba(). "
            "Use a classifier trained with probability support."
        )

    probabilities = model.predict_proba(input_df)
    classes = getattr(model, "classes_", None)

    if classes is not None:
        positive_indices = [
            i for i, label in enumerate(classes)
            if str(label).strip().lower() in {"1", "1.0", "true", "yes", "positive"}
        ]

        if positive_indices:
            positive_index = positive_indices[0]
        elif len(classes) == 2:
            positive_index = 1
        else:
            raise ValueError(
                f"Unable to identify the positive class. Classes: {classes}"
            )
    else:
        if probabilities.shape[1] != 2:
            raise ValueError(
                "Unable to identify the positive class in model output."
            )
        positive_index = 1

    return float(probabilities[0][positive_index])


def classify_risk(probability):
    """Convert probability to a simple display category."""
    if probability < 0.30:
        return "Lower", "pg-low"

    if probability < 0.60:
        return "Moderate", "pg-moderate"

    return "Elevated", "pg-high"


# ==============================================================================
# DYNAMIC RULE-BASED CLINICAL RECOMMENDATION ENGINE
# ==============================================================================
def generate_health_tips(age, sex, heart_prob, diab_prob, trestbps, chol, glucose, bmi, fbs_val):
    """
    Generates tailored, factor-specific clinical action plans based on individual biometrics.
    """
    diet_steps = []
    diagnostic_tests = []
    lifestyle_goals = []

    # 1. Blood Pressure Factors
    if trestbps >= 140:
        diet_steps.append("**High BP Alert (≥140 mmHg):** Strictly limit sodium intake (< 1,500 mg/day). Avoid processed/canned foods, papads, and added table salt.")
        diagnostic_tests.append("**Blood Pressure Monitoring:** Record resting BP twice daily (morning & evening) for 7 consecutive days and share logs with a physician.")
    elif trestbps >= 120:
        diet_steps.append("**Pre-Hypertension Guard (120-139 mmHg):** Adopt a DASH-style diet rich in potassium (spinach, bananas, coconut water) and cut back on sodium.")
        diagnostic_tests.append("**Follow-up:** Schedule a re-check of resting blood pressure in 10–14 days.")
    else:
        diet_steps.append("**Optimal BP:** Maintain balanced salt and hydration levels.")

    # 2. Cholesterol Factors
    if chol >= 240:
        diet_steps.append("**High Cholesterol Alert (≥240 mg/dL):** Eliminate trans fats, deep-fried snacks, and saturated palm oils. Increase soluble fiber (oats, flaxseeds, legumes).")
        diagnostic_tests.append("**Lipid Profile:** Schedule a comprehensive Fasting Lipid Panel (HDL, LDL, Triglycerides, VLDL) at the clinic.")
    elif chol >= 200:
        diet_steps.append("**Borderline High Cholesterol:** Limit saturated fats, butter, ghee, and full-fat dairy cream.")
    else:
        diet_steps.append("**Optimal Lipids:** Maintain current healthy fat balance (nuts, seeds, olive/mustard oil in moderation).")

    # 3. Blood Glucose Factors
    if glucose >= 140 or fbs_val == 1:
        diet_steps.append("**Elevated Glycemic Alert (≥140 mg/dL):** Strictly avoid refined sugars, sweets, fruit juices, white flour (maida), and high-GI carbohydrates.")
        diagnostic_tests.append("**HbA1c Blood Panel:** Schedule a 3-month Glycated Hemoglobin (HbA1c) fasting blood test within 3–5 days for long-term glycemic validation.")
    elif glucose >= 100:
        diet_steps.append("**Borderline Fasting Glucose (100-139 mg/dL):** Practice strict portion control for complex carbs (whole wheat, brown rice) balanced with lean protein.")
    else:
        diet_steps.append("**Optimal Glycemic Profile:** Fasting sugar levels are within ideal physiological limits.")

    # 4. Body Mass Index Factors
    if bmi >= 30:
        lifestyle_goals.append("**Obesity Management (BMI ≥ 30):** Target a structured, progressive 5–10% body weight reduction over 3–6 months under medical supervision.")
    elif bmi >= 25:
        lifestyle_goals.append("**Overweight Alert (BMI 25-29.9):** Limit caloric intake from processed snacks, sugary beverages, and late-night heavy suppers.")
    elif bmi < 18.5:
        lifestyle_goals.append("**Underweight Range (BMI < 18.5):** Focus on calorie-dense, protein-rich whole foods to achieve healthy muscle mass.")
    else:
        lifestyle_goals.append("**Healthy Weight Range:** Maintain current physical activity and balanced daily caloric intake.")

    # 5. Model Risk Escalation Actions
    if heart_prob >= 60:
        diagnostic_tests.append("🚨 **CRITICAL CARDIAC TRIAGE (Risk ≥60%):** Immediate clinical consultation and 12-lead Electrocardiogram (ECG) / Treadmill Test (TMT) recommended.")
    elif heart_prob >= 35:
        diagnostic_tests.append("⚠️ **MODERATE CARDIAC RISK:** Schedule a routine cardiology evaluation within 14 days.")

    if diab_prob >= 60:
        diagnostic_tests.append("🚨 **CRITICAL METABOLIC TRIAGE (Risk ≥60%):** Requires prompt clinical assessment for medical glycemic intervention.")

    if age >= 50:
        lifestyle_goals.append("**Senior Wellness Care (50+ Years):** Incorporate low-impact joint mobility exercises and schedule annual comprehensive health checkups.")

    # 6. Physical Activity Triage
    if heart_prob < 60:
        lifestyle_goals.append("**Aerobic Exercise Protocol:** Target at least 150 minutes of moderate-intensity activity (e.g., brisk walking, cycling) per week.")
    else:
        lifestyle_goals.append("⚠️ **Exercise Clearance Notice:** Begin physical exercises only after receiving explicit clearance from a medical doctor.")

    diet_fmt = "\n".join([f"- {item}" for item in diet_steps])
    diag_fmt = "\n".join([f"- {item}" for item in diagnostic_tests])
    life_fmt = "\n".join([f"- {item}" for item in lifestyle_goals])

    return diet_fmt, diag_fmt, life_fmt


def render_risk_card(title, probability, description):
    """Render a consistent risk result card."""
    risk_name, risk_class = classify_risk(probability)
    percent = probability * 100

    st.markdown(
        f"""
        <div class="pg-card">
            <div class="pg-card-title">{title}</div>
            <div class="pg-muted">{description}</div>
            <div class="pg-risk {risk_class}">{risk_name} risk</div>
            <div style="font-size: 25px; font-weight: 700; color: #102A43;">
                {percent:.1f}%
            </div>
            <div class="pg-muted">
                Model-estimated probability for the positive class
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR — VISIBLE, LABELED INPUTS
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 PulseGuard AI")
    st.markdown("### Clinical Screening")
    st.markdown(
        "Enter the available health information below to generate "
        "model-based screening estimates."
    )

    st.divider()

    st.markdown("### ❤️ Heart Health Inputs")

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=45,
        step=1,
        help="Enter age in years."
    )

    sex_label = st.selectbox(
        "Sex",
        options=["Male", "Female"],
        index=0
    )
    sex = 1 if sex_label == "Male" else 0

    cp_label = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic"
        ],
        index=0
    )
    cp = [
        "Typical angina",
        "Atypical angina",
        "Non-anginal pain",
        "Asymptomatic"
    ].index(cp_label)

    trestbps = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )

    chol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=50,
        max_value=700,
        value=200,
        step=1
    )

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=["No", "Yes"],
        index=0
    )
    fbs = 1 if fbs_label == "Yes" else 0

    restecg_label = st.selectbox(
        "Resting ECG Result",
        options=[
            "Normal",
            "ST-T wave abnormality",
            "Left ventricular hypertrophy"
        ],
        index=0
    )
    restecg = [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ].index(restecg_label)

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

    exang_label = st.selectbox(
        "Exercise-Induced Angina",
        options=["No", "Yes"],
        index=0
    )
    exang = 1 if exang_label == "Yes" else 0

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f"
    )

    slope_label = st.selectbox(
        "ST Segment Slope",
        options=[
            "Upsloping",
            "Flat",
            "Downsloping"
        ],
        index=0
    )
    slope = [
        "Upsloping",
        "Flat",
        "Downsloping"
    ].index(slope_label)

    ca = st.selectbox(
        "Major Vessels (CA)",
        options=[0, 1, 2, 3, 4],
        index=0
    )

    thal_label = st.selectbox(
        "Thalassemia (Thal)",
        options=[
            "Normal",
            "Fixed defect",
            "Reversible defect"
        ],
        index=0
    )
    thal = [
        "Normal",
        "Fixed defect",
        "Reversible defect"
    ].index(thal_label)

    st.divider()

    st.markdown("### 🩸 Diabetes Inputs")

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=30,
        value=0,
        step=1
    )

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0,
        max_value=500,
        value=110,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mmHg) — Diabetes Model",
        min_value=0,
        max_value=250,
        value=72,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    insulin = st.number_input(
        "Insulin (mu U/mL)",
        min_value=0,
        max_value=1000,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        format="%.1f"
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.2f"
    )

    diabetes_age = st.number_input(
        "Age (years) — Diabetes Model",
        min_value=1,
        max_value=120,
        value=35,
        step=1
    )

    st.divider()

    run_screening = st.button(
        "🔎 Run Health Screening",
        use_container_width=True
    )


# =========================================================
# MAIN DASHBOARD HEADER
# =========================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #102A43, #176B87);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 22px;
    ">
        <div style="
            font-size: 14px;
            color: #BFE8F0;
            font-weight: 600;
            letter-spacing: 1px;
        ">
            AI-ASSISTED HEALTH SCREENING
        </div>
        <div style="
            font-size: 34px;
            color: #FFFFFF;
            font-weight: 800;
            margin-top: 5px;
        ">
            PulseGuard AI
        </div>
        <div style="
            font-size: 16px;
            color: #FFFFFF;
            margin-top: 8px;
        ">
            Understand model-estimated heart disease and diabetes risk
            using the health information you provide.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## Your Health Dashboard")

st.info(
    "This application is an educational screening prototype. "
    "Its predictions are not medical diagnoses and must not replace "
    "evaluation by a qualified healthcare professional."
)


# =========================================================
# BUILD MODEL INPUT DATAFRAMES
# =========================================================

heart_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

heart_input = pd.DataFrame(
    [[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]],
    columns=heart_columns
)

diabetes_columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

diabetes_input = pd.DataFrame(
    [[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, diabetes_age
    ]],
    columns=diabetes_columns
)


# =========================================================
# RESULTS & CUSTOM CLINICAL ACTION PLAN CARDS
# =========================================================

if run_screening:

    heart_probability = None
    diabetes_probability = None

    heart_error = None
    diabetes_error = None

    if heart_model is not None:
        try:
            heart_probability = get_positive_class_probability(
                heart_model,
                heart_input
            )
        except Exception as exc:
            heart_error = str(exc)

    if diabetes_model is not None:
        try:
            diabetes_probability = get_positive_class_probability(
                diabetes_model,
                diabetes_input
            )
        except Exception as exc:
            diabetes_error = str(exc)

    if heart_error:
        st.error(f"Heart model prediction failed: {heart_error}")

    if diabetes_error:
        st.error(f"Diabetes model prediction failed: {diabetes_error}")

    if heart_probability is not None or diabetes_probability is not None:

        st.markdown("### Screening Results")

        col1, col2 = st.columns(2)

        with col1:
            if heart_probability is not None:
                render_risk_card(
                    "❤️ Heart Disease",
                    heart_probability,
                    "Estimated probability from the heart disease model."
                )
            else:
                st.warning("Heart disease result is unavailable.")

        with col2:
            if diabetes_probability is not None:
                render_risk_card(
                    "🩸 Diabetes",
                    diabetes_probability,
                    "Estimated probability from the diabetes model."
                )
            else:
                st.warning("Diabetes result is unavailable.")

        st.markdown("### Input Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Heart Age", f"{age} years")

        with summary_col2:
            st.metric("Blood Pressure", f"{trestbps} mmHg")

        with summary_col3:
            st.metric("BMI", f"{bmi:.1f}")

        # Calculate percentages for the health tips generator
        h_prob_percent = (heart_probability * 100) if heart_probability is not None else 0
        d_prob_percent = (diabetes_probability * 100) if diabetes_probability is not None else 0

        # Generate Factor-Specific Clinical Suggestions
        diet_fmt, diag_fmt, life_fmt = generate_health_tips(
            age, sex_label, h_prob_percent, d_prob_percent, trestbps, chol, glucose, bmi, fbs
        )

        st.markdown("### 💡 Personalized Clinical Guidance & Action Plan")

        # --- CARD 1: DIET & NUTRITION ---
        st.markdown("""
        <div style="
            background-color: #FFFFFF;
            border-left: 5px solid #10B981;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 16px; font-weight: 700; color: #065F46; margin-bottom: 10px;">
                🍏 Dietary & Nutrition Protocol
            </div>
            <div style="color: #1F2937; font-size: 14px; line-height: 1.6;">
        """, unsafe_allow_html=True)
        st.markdown(diet_fmt)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # --- CARD 2: DIAGNOSTIC TESTS ---
        st.markdown("""
        <div style="
            background-color: #FFFFFF;
            border-left: 5px solid #0284C7;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 16px; font-weight: 700; color: #075985; margin-bottom: 10px;">
                🔬 Recommended Clinical & Laboratory Tests
            </div>
            <div style="color: #1F2937; font-size: 14px; line-height: 1.6;">
        """, unsafe_allow_html=True)
        st.markdown(diag_fmt)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # --- CARD 3: LIFESTYLE & GOALS ---
        st.markdown("""
        <div style="
            background-color: #FFFFFF;
            border-left: 5px solid #F59E0B;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 16px; font-weight: 700; color: #92400E; margin-bottom: 10px;">
                🏃 Lifestyle & Physical Activity Goals
            </div>
            <div style="color: #1F2937; font-size: 14px; line-height: 1.6;">
        """, unsafe_allow_html=True)
        st.markdown(life_fmt)
        st.markdown("</div></div>", unsafe_allow_html=True)

        with st.expander("View model input values"):
            st.markdown("**Heart disease model inputs**")
            st.dataframe(heart_input, use_container_width=True)

            st.markdown("**Diabetes model inputs**")
            st.dataframe(diabetes_input, use_container_width=True)

    else:
        st.error(
            "No predictions were generated. Check that both model files "
            "exist and are compatible with these input features."
        )

else:
    # Initial dashboard state before the user runs screening
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="pg-card">
                <div class="pg-card-title">❤️ Heart Health</div>
                <div class="pg-muted">
                    Enter your heart health information in the sidebar,
                    then select Run Health Screening.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="pg-card">
                <div class="pg-card-title">🩸 Diabetes Screening</div>
                <div class="pg-muted">
                    Enter your diabetes-related information in the sidebar
                    to generate a model-based estimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### How to use PulseGuard AI")

    st.markdown("""
    1. Enter the available health information in the left sidebar.
    2. Check that the values are correct.
    3. Click **Run Health Screening**.
    4. Review the model-estimated probabilities and general health suggestions.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="pg-footer">
        PulseGuard AI · Educational screening prototype<br>
        Not a substitute for professional medical advice, diagnosis or treatment.
    </div>
    """,
    unsafe_allow_html=True
)