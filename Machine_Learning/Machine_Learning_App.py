import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Intelligence Models UI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        color: #f1f1f1;
        font-family: 'Inter', sans-serif;
    }
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #171724;
        border-right: 1px solid #33334d;
    }
    /* Cards / Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
    }
    /* Highlighted text */
    .highlight {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4ECDC4, #556270);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
    }
    /* Output Box */
    .output-box {
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ECDC4;
        padding: 20px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 20px;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'><span class='highlight'>Machine Learning Intelligence</span> Portal</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        height_model = pickle.load(open('height_intelligence.pkl', 'rb'))
        gender_model = pickle.load(open('gender_classification_intelligence.pkl', 'rb'))
        suv_model = pickle.load(open('suv_prediction_model.pkl', 'rb'))
        return height_model, gender_model, suv_model
    except Exception as e:
        st.error(f"Error loading models. Please make sure the .pkl files exist in the directory. Error: {e}")
        return None, None, None

height_model, gender_model, suv_model = load_models()

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 class='highlight'>Menu</h2>", unsafe_allow_html=True)
    st.markdown("Select a predictive model to use:")
    app_mode = st.radio("Choose Model", ["Weight Prediction", "Gender Classification", "SUV Purchase Prediction"])
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This portal serves serialized Machine Learning models from the repository for interactive testing.")

# Main Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if height_model and gender_model and suv_model:
        if app_mode == "Weight Prediction":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>🏋️ Weight Prediction Model</h3>
                <p style='color: #bbb;'>Predict a person's weight (kg) based on their height (cm) using a trained Linear Regression model.</p>
            </div>
            """, unsafe_allow_html=True)
            
            height_input = st.slider("Select Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
            
            if st.button("Predict Weight"):
                prediction = height_model.predict([[height_input]])
                predicted_weight = prediction[0][0] if isinstance(prediction[0], (list, np.ndarray)) else prediction[0]
                
                st.markdown(f"""
                <div class='output-box'>
                    <strong>Predicted Weight:</strong> <span style='font-size: 1.5em; color: #4ECDC4;'>{predicted_weight:.2f} kg</span>
                </div>
                """, unsafe_allow_html=True)
                
        elif app_mode == "Gender Classification":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>🚻 Gender Classification Model</h3>
                <p style='color: #bbb;'>Classify a person's gender based on their height (cm) and weight (kg) using Logistic Regression.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                h_input = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
            with c2:
                w_input = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0, step=0.1)
                
            if st.button("Classify Gender"):
                prediction = gender_model.predict([[h_input, w_input]])
                predicted_gender = prediction[0]
                
                prob = gender_model.predict_proba([[h_input, w_input]])
                confidence = max(prob[0]) * 100
                
                color = "#FF6B6B" if predicted_gender == "Female" else "#4ECDC4"
                icon = "👩" if predicted_gender == "Female" else "👨"
                
                st.markdown(f"""
                <div class='output-box' style='border-left-color: {color}; background: rgba(255, 255, 255, 0.05);'>
                    <strong>Predicted Gender:</strong> <span style='font-size: 1.5em; color: {color};'>{icon} {predicted_gender}</span><br>
                    <small style='color: #888;'>Confidence: {confidence:.2f}%</small>
                </div>
                """, unsafe_allow_html=True)
                
        elif app_mode == "SUV Purchase Prediction":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>🚙 SUV Purchase Prediction</h3>
                <p style='color: #bbb;'>Predict whether a user will purchase an SUV based on their Age and Estimated Salary.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                age_input = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            with c2:
                salary_input = st.number_input("Estimated Salary ($)", min_value=15000, max_value=200000, value=50000, step=1000)
                
            if st.button("Predict Purchase"):
                prediction = suv_model.predict([[age_input, salary_input]])
                prob = suv_model.predict_proba([[age_input, salary_input]])
                
                purchased = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"
                confidence = max(prob[0]) * 100
                
                color = "#4ECDC4" if prediction[0] == 1 else "#FF6B6B"
                icon = "✅" if prediction[0] == 1 else "❌"
                
                st.markdown(f"""
                <div class='output-box' style='border-left-color: {color}; background: rgba(255, 255, 255, 0.05);'>
                    <strong>Prediction:</strong> <span style='font-size: 1.5em; color: {color};'>{icon} {purchased}</span><br>
                    <small style='color: #888;'>Confidence: {confidence:.2f}%</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Models are not loaded correctly.")
