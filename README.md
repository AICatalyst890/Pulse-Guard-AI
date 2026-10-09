# 🩺 PulseGuard AI: Early Chronic Disease Risk Assessment & Clinical Triage Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange.svg)](https://scikit-learn.org/)
[![SDG Target](https://img.shields.io/badge/SDG-3%3A%20Good%20Health%20%26%20Well--Being-green.svg)](https://sdgs.un.org/goals/goal3)

PulseGuard AI is an intelligent clinical decision support system designed for non-specialist healthcare workers and nurses in rural or low-resource primary healthcare centers (PHCs). By leveraging dual Supervised Machine Learning algorithms and a factor-specific dynamic clinical triage engine, PulseGuard AI enables early risk screening for **Cardiovascular Disease** and **Type-2 Diabetes**, aligning directly with **UN Sustainable Development Goal 3 (SDG 3: Good Health and Well-Being)**.

---

## 📌 Project Features

- **Dual-Disease ML Analytics Engine:** Evaluates patient biometrics simultaneously across two separate machine learning classification pipelines.
- **Cardiovascular Disease Risk Pipeline:** Trained using Scikit-Learn pipelines to predict cardiac risk probability.
- **Diabetes & Metabolic Risk Pipeline:** Predicts metabolic disorder risk based on standard clinical parameters (Glucose, BMI, Insulin, Age, etc.).
- **Dynamic Factor-Specific Health Guidance:** Generates instant clinical triage recommendations, dietary plans, required diagnostic follow-ups, and lifestyle goals based on individual patient thresholds.
- **User-Friendly Dashboard:** Responsive web interface built with Streamlit for rapid biometric data entry by healthcare workers.

---

## 📁 Repository Structure

```text
pulseguard-ai/
├── data/                       # Raw clinical datasets
│   ├── processed.cleveland.data
│   └── diabetes.csv
├── models/                     # Saved Scikit-Learn trained pipeline artifacts
│   ├── heart_model.pkl
│   └── diabetes_model.pkl
├── app.py                      # Main Streamlit web application
├── train_models.ipynb          # Model training, preprocessing, & evaluation notebook
├── requirements.txt            # Project Python dependencies
└── README.md                   # Project documentation

```

---

## 🛠️ Installation & Local Setup

### Prerequisites

* Python 3.9+ installed on your system.
* Git installed.

### Step 1: Clone the Repository

```bash
git clone https://github.com/AICatalyst890/Pulse-Guard-AI.git
cd Pulse-Guard-AI

```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

```

### Step 3: Launch the Streamlit Web Application

```bash
streamlit run app.py

```

Open your web browser and navigate to `http://localhost:8501`.

---

## 📊 Model Architecture & Performance

The models were trained inside `train_models.ipynb` using supervised machine learning algorithms with automated feature scaling and preprocessing pipelines:

1. **Heart Disease Model (`heart_model.pkl`):** Trained on the Cleveland Heart Disease Dataset.
2. **Diabetes Model (`diabetes_model.pkl`):** Trained on the PIMA Indian Diabetes Dataset.

---

## 🎯 Alignment with Sustainable Development Goals (SDG 3)

PulseGuard AI addresses the shortage of specialist doctors in low-resource clinics by empowering primary healthcare workers with early screening tools, reducing delayed diagnoses for Non-Communicable Diseases (NCDs).

```