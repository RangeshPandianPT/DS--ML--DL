import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        iris_model = pickle.load(open('iris_classification_model.pkl', 'rb'))
        wine_model = pickle.load(open('wine_quality_model.pkl', 'rb'))
        stock_model = pickle.load(open('stock_prediction_model.pkl', 'rb'))
        return height_model, gender_model, suv_model, iris_model, wine_model, stock_model
    except Exception as e:
        st.error(f"Error loading models. Please make sure the .pkl files exist in the directory. Error: {e}")
        return None, None, None, None, None, None

height_model, gender_model, suv_model, iris_model, wine_model, stock_model = load_models()

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 class='highlight'>Menu</h2>", unsafe_allow_html=True)
    st.markdown("Select an application mode:")
    app_mode = st.radio("Choose Mode", ["Weight Prediction", "Gender Classification", "SUV Purchase Prediction", "Iris Species Classification", "Wine Quality Prediction", "Stock Price Prediction", "Data Explorer"])
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This portal serves serialized Machine Learning models from the repository for interactive testing.")

# Main Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if all(m is not None for m in [height_model, gender_model, suv_model, iris_model, wine_model, stock_model]):
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
                
        elif app_mode == "Iris Species Classification":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>🌸 Iris Species Classification</h3>
                <p style='color: #bbb;'>Predict the species of an Iris flower based on its measurements.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                sl = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
                pl = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            with c2:
                sw = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                pw = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
                
            if st.button("Predict Species"):
                prediction = iris_model.predict([[sl, sw, pl, pw]])
                species = prediction[0]
                
                st.markdown(f"""
                <div class='output-box' style='border-left-color: #FFD166; background: rgba(255, 255, 255, 0.05);'>
                    <strong>Predicted Species:</strong> <span style='font-size: 1.5em; color: #FFD166;'>{species}</span>
                </div>
                """, unsafe_allow_html=True)

        elif app_mode == "Wine Quality Prediction":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>🍷 Wine Quality Prediction</h3>
                <p style='color: #bbb;'>Predict the quality of a wine (score between 0 and 10) based on chemical properties.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                fa = st.number_input("Fixed Acidity", value=7.4)
                rs = st.number_input("Residual Sugar", value=1.9)
                tsd = st.number_input("Total Sulfur Dioxide", value=34.0)
                sul = st.number_input("Sulphates", value=0.56)
            with c2:
                va = st.number_input("Volatile Acidity", value=0.7)
                ch = st.number_input("Chlorides", value=0.076)
                den = st.number_input("Density", value=0.9978)
                alc = st.number_input("Alcohol", value=9.4)
            with c3:
                ca = st.number_input("Citric Acid", value=0.0)
                fsd = st.number_input("Free Sulfur Dioxide", value=11.0)
                ph = st.number_input("pH", value=3.51)
                
            if st.button("Predict Quality"):
                features = [[fa, va, ca, rs, ch, fsd, tsd, den, ph, sul, alc]]
                prediction = wine_model.predict(features)
                quality = prediction[0]
                
                st.markdown(f"""
                <div class='output-box' style='border-left-color: #EF476F; background: rgba(255, 255, 255, 0.05);'>
                    <strong>Predicted Quality Score:</strong> <span style='font-size: 1.5em; color: #EF476F;'>{quality:.2f} / 10</span>
                </div>
                """, unsafe_allow_html=True)

        elif app_mode == "Stock Price Prediction":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>📈 Stock Price Prediction</h3>
                <p style='color: #bbb;'>Predict the closing price of a stock based on its daily metrics.</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                o_price = st.number_input("Open Price ($)", value=39.69)
                l_price = st.number_input("Low Price ($)", value=38.79)
            with c2:
                h_price = st.number_input("High Price ($)", value=41.22)
                vol = st.number_input("Volume", value=24232729)
                
            if st.button("Predict Close Price"):
                prediction = stock_model.predict([[o_price, h_price, l_price, vol]])
                close_price = prediction[0]
                
                st.markdown(f"""
                <div class='output-box' style='border-left-color: #118AB2; background: rgba(255, 255, 255, 0.05);'>
                    <strong>Predicted Close Price:</strong> <span style='font-size: 1.5em; color: #118AB2;'>${close_price:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
        elif app_mode == "Data Explorer":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='margin-bottom: 20px;'>📊 Interactive Data Explorer</h3>
                <p style='color: #bbb;'>Explore the datasets used to train these models.</p>
            </div>
            """, unsafe_allow_html=True)
            
            dataset_choice = st.selectbox("Choose a dataset", ["Iris", "Wine Quality", "Stock Data"])
            
            if dataset_choice == "Iris":
                df = pd.read_csv("Iris.csv")
                st.dataframe(df.head(10))
                st.markdown("**Scatter Plot (Sepal Length vs Sepal Width)**")
                st.scatter_chart(data=df, x='SepalLengthCm', y='SepalWidthCm', color='Species')
            elif dataset_choice == "Wine Quality":
                df = pd.read_csv("WineQT.csv")
                st.dataframe(df.head(10))
                st.markdown("**Histogram of Quality**")
                st.bar_chart(df['quality'].value_counts())
            elif dataset_choice == "Stock Data":
                df = pd.read_csv("stock_data.csv")
                # Drop rows with NaN in Date or Close
                df = df.dropna(subset=['Date', 'Close'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                st.dataframe(df.head(10))
                st.markdown("**Closing Price Over Time**")
                st.line_chart(df['Close'])

    else:
        st.warning("Models are not loaded correctly. Please run the training scripts to generate all .pkl files.")
