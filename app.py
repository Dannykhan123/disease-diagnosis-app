import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("disease_diagnosis_model.pkl")

# Load symptom list from the trained model
symptom_list = list(model.feature_names_in_)

# Set Streamlit page layout
st.set_page_config(page_title="AI Disease Diagnosis", page_icon="🩺", layout="wide")

# Custom CSS for background image and UI styling
st.markdown(
    """
    <style>
        /* Apply Background Image */
        .stApp {
            background: url("https://www.publicdomainpictures.net/pictures/280000/velka/medical-background-1543241046hPq.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        /* Page Title */
        .title {
            color: #ffffff;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            padding: 10px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
        }
        /* Section Headers */
        .section {
            color: #ffcc00;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        /* Button Styling */
        .stButton>button {
            color: white;
            background-color: #007bff;
            font-size: 18px;
            padding: 8px 20px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        /* Prediction Box */
        .result-box {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: black;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Title with Styling
st.markdown("<p class='title'>🩺 AI-Powered Disease Diagnosis System</p>", unsafe_allow_html=True)

# User Instruction
st.write("🔍 Select symptoms to get a disease prediction.")

# Symptom selection in dropdown
selected_symptoms = st.multiselect(
    "🔍 Search & Select Symptoms",
    options=symptom_list,
    default=[],
)

# Create two columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    st.write("")  # Placeholder for alignment

with col2:
    if st.button("🧪 Predict Disease", help="Click to diagnose based on selected symptoms"):
        if selected_symptoms:
            # Convert user input into model format
            user_input = np.zeros(len(symptom_list))
            for symptom in selected_symptoms:
                user_input[symptom_list.index(symptom)] = 1

            # Make prediction
            prediction = model.predict([user_input])[0]

            # Display the result in a styled box
            st.markdown(f"<div class='result-box'>🎯 **Predicted Disease:** {prediction}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select at least one symptom to proceed.")
