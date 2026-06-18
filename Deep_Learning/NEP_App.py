import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Set page config
st.set_page_config(page_title="Name Gender Predictor", page_icon="🚻", layout="centered")

st.title("🚻 Name Gender Predictor")
st.write("This app uses a Deep Learning LSTM model to predict the gender associated with a name.")

@st.cache_resource
def load_nep_model():
    model_path = 'saved_models/nep_model.keras'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = load_nep_model()

if model is None:
    st.error("Model not found! Please run the NEP.ipynb notebook to train and save the model to 'saved_models/nep_model.keras'.")
else:
    # Prepare vocabulary
    vocab = {chr(i + 96): i for i in range(1, 27)}
    MAX_LEN = 10
    
    # User Input
    name_input = st.text_input("Enter a Name:", "")
    
    if st.button("Predict Gender"):
        if name_input.strip() == "":
            st.warning("Please enter a valid name.")
        else:
            test_name = name_input.strip().lower()
            
            # Preprocess
            seq = [vocab[ch] for ch in test_name if ch in vocab]
            x_test = pad_sequences([seq], maxlen=MAX_LEN, padding='pre')
            
            # Predict
            prediction = model.predict(x_test)[0][0]
            
            # Display results
            st.subheader("Result:")
            if prediction < 0.5:
                st.success(f"👧 **{name_input.capitalize()}** is predicted to be **Female**.")
                st.info(f"Confidence score (closer to 0 is female): {prediction:.4f}")
            else:
                st.info(f"👦 **{name_input.capitalize()}** is predicted to be **Male**.")
                st.info(f"Confidence score (closer to 1 is male): {prediction:.4f}")
