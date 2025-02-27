import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from gtts import gTTS
import os
import base64
import tempfile

# Load the trained model
MODEL_PATH = "currency_classifier.h5"  # Change this if needed
model = load_model(MODEL_PATH, compile=False)  # Prevent loading old optimizer
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # Recompile

# Define class labels (map indices to class names)
class_labels = {0: 'Rs. 10', 1: 'Rs. 100', 2: 'Rs. 20', 3: 'Rs. 200',
                4: 'Rs. 2000', 5: 'Rs. 50', 6: 'Rs. 500', 7: 'Nothing only background is seen'}

# Streamlit UI
st.title("Indian Currency Note Classification")
st.write("Upload an image of an Indian currency note to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def predict_image(img):
    img = img.resize((64, 64))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch dimension
    img_array = img_array / 255.0  # Normalize
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get class with highest probability
    confidence = np.max(predictions)  # Get confidence score
    
    return class_labels[predicted_class], confidence

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_filename = temp_audio.name
        tts.save(temp_filename)

    # Convert audio to base64 so it can be played directly
    with open(temp_filename, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    # Autoplay JavaScript (ensures audio plays automatically)
    audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

    # Remove temp file after encoding
    os.remove(temp_filename)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Predict the class
    label, confidence = predict_image(img)
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
    
    # Convert prediction to speech and autoplay
    speech_text = f"The detected currency note is {label} with a confidence of {confidence:.2f}"
    text_to_speech(speech_text)
