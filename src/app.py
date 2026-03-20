import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Load pre-trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(os.path.join(BASE_DIR, 'models', 'custom_best_model.h5'))
# Set image size
im_size = 224

# Prediction function
def predict_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (im_size, im_size)) / 255.0
    tensor_image = np.expand_dims(img, axis=0)
    prediction = model.predict(tensor_image, verbose=0).round(3)
    return prediction

# Streamlit interface
st.title("Wildfire Prediction Interface")
st.write("Upload an image to predict wildfire probability.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict_image(uploaded_file)
    st.write("Prediction Results:")
    st.write(f"No Wildfire Probability: {prediction[0][0] * 100:.2f}%")
    st.write(f"Wildfire Probability: {prediction[0][1] * 100:.2f}%")