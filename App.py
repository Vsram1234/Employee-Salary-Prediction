import streamlit as st
import pandas as pd
import joblib
import os

# ==========================
# Page Config & Style
# ==========================
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="wide")

# Custom CSS for overall app style
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
    }

    /* Title styling */
    h1 {
        font-size: 2.5rem !important;
        color: #00695c !important;
        text-align: center;
        margin-bottom: 1rem;
    }

    h2, h3, h4 {
        color: #004d40 !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        border: 2px solid #004d40;
        border-radius: 10px;
        padding: 10px;
    }

    /* Prediction box styling */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: white;
        margin-top: 1rem;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    .prediction-high { background-color: #2e7d32; }  /* Green */
    .prediction-low { background-color: #c62828; }   /* Red */

    /* Tabs and sections */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #80cbc4;
        border-radius: 5px;
        color: white !important;
        font-weight: bold;
        padding: 8px 12px;
    }
    .stTabs [aria-selected="true"] {
        background: #004d40;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Employee Salary Prediction Dashboard")
st.markdown("<h3 style='text-align:center;'>Predict salaries and analyze model performance</h3>", unsafe_allow_html=True)

# ==========================
# Load trained pipeline/model
# ==========================
try:
    model = joblib.load("best_model.pkl")
except Exception:
    st.error("⚠️ Trained model not found. Please run the training script first.")
    st.stop()

# ==========================
# Sidebar: Employee Details
# ==========================
st.sidebar.header("📝 Enter Employee Details")
st.sidebar.markdown("Customize employee profile to predict salary category.")

# Dropdown options
workclass_options = ['Private', 'Self-emp', 'Government', 'Others']
marital_status_options = ['Married', 'Single', 'Divorced', 'Widowed']
occupation_options = ['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Others']
relationship_options = ['Husband', 'Wife', 'Not-in-family', 'Unmarried', 'Other']
race_options = ['White', 'Black', 'Asian', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'India', 'Other']

# Inputs
age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass", workclass_options)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
occupation = st.sidebar.selectbox("Occupation", occupation_options)
relationship = st.sidebar.selectbox("Relationship", relationship_options)
race = st.sidebar.selectbox("Race", race_options)
gender = st.sidebar.selectbox("Gender", gender_options)
native_country = st.sidebar.selectbox("Native Country", native_country_options)
educational_num = st.sidebar.slider("Educational Number (5–16)", 5, 16, 10)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

# ==========================
# Prepare Input Data
# ==========================
input_df = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "gender": gender,
    "native-country": native_country,
    "educational-num": educational_num,
    "capital-gain": capital_gain,
    "hours-per-week": hours_per_week
}])

st.subheader("🔍 Your Input Data")
st.dataframe(input_df, use_container_width=True)

# ==========================
# Prediction Section
# ==========================
if st.button("🚀 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    result_class = ">50K" if prediction == 1 else "≤50K"
    css_class = "prediction-high" if prediction == 1 else "prediction-low"
    st.markdown(f'<div class="prediction-box {css_class}">Prediction: {result_class}</div>', unsafe_allow_html=True)

# ==========================
# Batch Prediction
# ==========================
st.markdown("---")
st.subheader("📂 Batch Salary Prediction")
uploaded_file = st.file_uploader("Upload a CSV (with same columns as dataset) for batch prediction", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = ['>50K' if p == 1 else '≤50K' for p in batch_preds]
    st.write("### Sample Predictions", batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Predictions",
        csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

# ==========================
# Model Performance Visuals
# ==========================
st.markdown("---")
st.header("📊 Model Performance")

PLOTS_DIR = "model_graphs"
if os.path.exists(PLOTS_DIR):
    # Tabs for better navigation
    tab1, tab2 = st.tabs(["Accuracy Comparison", "Detailed Metrics"])
    
    with tab1:
        st.image(os.path.join(PLOTS_DIR, "accuracy_comparison.png"), caption="Model Accuracy Comparison", use_container_width=True)
    
    with tab2:
        cols = st.columns(2)
        for plot in sorted(os.listdir(PLOTS_DIR)):
            if plot.endswith(".png") and plot != "accuracy_comparison.png":
                with cols[0] if "confusion" in plot else cols[1]:
                    st.image(os.path.join(PLOTS_DIR, plot))
else:
    st.warning("⚠️ No performance graphs found. Run the training script to generate them.")
