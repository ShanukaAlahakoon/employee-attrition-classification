import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Model Loading with st.cache_resource ---
# Using st.cache_resource to load the model once and cache it.
# This prevents reloading the model every time the app re-runs.

MODEL_FILENAME = 'best_model_employee_attition.joblib'
# NOTE: In a real deployment, you would need to ensure this file is
# present alongside the streamlit_app.py or downloaded.

@st.cache_resource
def load_model_pipeline(filename):
    """Loads the pre-trained scikit-learn pipeline."""
    if not os.path.exists(filename):
        # This is a placeholder for deployment: the user would need to ensure the file exists.
        st.error(f"Model file '{filename}' not found. Please upload or ensure it is in the correct path.")
        return None
    try:
        pipeline = joblib.load(filename)
        return pipeline
    except Exception as e:
        st.error(f"Error loading the model pipeline: {e}")
        return None

# Load the model
model_pipeline = load_model_pipeline(MODEL_FILENAME)

# Check if model loaded successfully before proceeding
if model_pipeline is None:
    st.stop() # Stop the app if the model didn't load

# --- Streamlit App Configuration and Title ---
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Employee Attrition Prediction App 📊")
st.markdown("Enter the employee's details to predict their likelihood of leaving the company (`LeaveOrNot=1`).")

# --- Define Feature Options based on Notebook Analysis ---
# Inferred from df.head() and df.info() in the notebook
EDUCATION_OPTIONS = ["Bachelors", "Masters", "PHD"]
CITY_OPTIONS = ["Bangalore", "Pune", "New Delhi"]
GENDER_OPTIONS = ["Male", "Female"]
EVERBENCHED_OPTIONS = ["Yes", "No"]
MIN_YEAR = 2012 # Inferred from df.describe() min
MAX_YEAR = 2018 # Inferred from df.describe() max
MIN_AGE = 22 # Inferred from df.describe() min
MAX_AGE = 41 # Inferred from df.describe() max
MIN_EXP = 0 # Inferred from df.describe() min
MAX_EXP = 7 # Inferred from df.describe() max
PAYMENT_TIER_OPTIONS = [1, 2, 3] # Inferred from df.head() and df.describe() unique values

# --- User Input Widgets ---

# Use a two-column layout for a cleaner look
col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "Education Level",
        options=EDUCATION_OPTIONS,
        index=EDUCATION_OPTIONS.index("Masters"),
        help="Highest level of education."
    )
    joining_year = st.number_input(
        "Joining Year",
        min_value=MIN_YEAR,
        max_value=MAX_YEAR,
        value=2017,
        step=1,
        help=f"The year the employee joined the company (Range: {MIN_YEAR}-{MAX_YEAR})."
    )
    city = st.selectbox(
        "City",
        options=CITY_OPTIONS,
        index=CITY_OPTIONS.index("New Delhi"),
        help="City where the employee is currently based."
    )
    payment_tier = st.selectbox(
        "Payment Tier",
        options=PAYMENT_TIER_OPTIONS,
        index=PAYMENT_TIER_OPTIONS.index(2),
        help="The employee's payment tier (1=Highest, 3=Lowest)."
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=MIN_AGE,
        max_value=MAX_AGE,
        value=27,
        step=1,
        help=f"Employee's age (Range: {MIN_AGE}-{MAX_AGE})."
    )
    gender = st.selectbox(
        "Gender",
        options=GENDER_OPTIONS,
        index=GENDER_OPTIONS.index("Female")
    )
    ever_benched = st.selectbox(
        "Ever Benched",
        options=EVERBENCHED_OPTIONS,
        index=EVERBENCHED_OPTIONS.index("No"),
        help="Indicates if the employee has ever been benched (Yes/No)."
    )
    experience_in_current_domain = st.number_input(
        "Experience in Current Domain (Years)",
        min_value=MIN_EXP,
        max_value=MAX_EXP,
        value=5,
        step=1,
        help=f"Years of experience in the current project domain (Range: {MIN_EXP}-{MAX_EXP})."
    )

# --- Prediction Logic ---
if st.button("Predict Attrition"):
    # 1. Collect inputs into a dictionary
    input_data = {
        "Education": education,
        "JoiningYear": joining_year,
        "City": city,
        "PaymentTier": payment_tier,
        "Age": age,
        "Gender": gender,
        "EverBenched": ever_benched,
        "ExperienceInCurrentDomain": experience_in_current_domain
    }

    # 2. Convert dictionary to a Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Make Prediction (0=Stay, 1=Leave)
    try:
        prediction = model_pipeline.predict(input_df)[0]
        # Get probabilities for both classes (Stay/Leave)
        probabilities = model_pipeline.predict_proba(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # 4. Interpret Results
    leave_probability = probabilities[1]
    stay_probability = probabilities[0]

    # Convert probability to percentage for display
    leave_confidence_percent = leave_probability * 100
    stay_confidence_percent = stay_probability * 100

    # Determine the outcome string and color/icon
    if prediction == 1:
        outcome_text = "**Leave** 😥 (Attrition Predicted)"
        st.error(f"**Prediction:** The employee is predicted to {outcome_text}.")
        st.metric(label="Confidence in 'Leave' Prediction", value=f"{leave_confidence_percent:.2f}%", delta_color="inverse")
    else:
        outcome_text = "**Stay** 😊 (No Attrition Predicted)"
        st.success(f"**Prediction:** The employee is predicted to {outcome_text}.")
        st.metric(label="Confidence in 'Stay' Prediction", value=f"{stay_confidence_percent:.2f}%", delta_color="normal")

    st.markdown("---")
    st.subheader("Probability Distribution")
    
    # Display the probabilities clearly
    prob_data = {
        'Outcome': ['Stay (0)', 'Leave (1)'],
        'Probability': [stay_probability, leave_probability]
    }
    prob_df = pd.DataFrame(prob_data)
    
    # Plotting probabilities (optional but good for visualization)
    st.bar_chart(prob_df.set_index('Outcome'))
    
    st.markdown("---")
    st.caption("Model used: Random Forest Classifier Pipeline")