import streamlit as st
import cv2
import numpy as np
import base64
import json
from deepface import DeepFace
import time
import matplotlib.pyplot as plt

def analyze_emotion(image):
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    return result[0]['emotion'] if result else {}

def normalize_scores(scores):
    total = sum(scores.values()) if scores else 1
    return {emotion: float((score / total) * 100) for emotion, score in scores.items()}

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump({k: float(v) for k, v in data.items()}, f)

st.title("Mood Tracker App")

# Sidebar for webcam controls
st.sidebar.header("Webcam Capture")
start_capture = st.sidebar.button("Start Capture")
stop_capture = st.sidebar.button("Stop Capture")

frame_placeholder = st.empty()
chart_placeholder = st.empty()

if start_capture:
    cap = cv2.VideoCapture(0)
    st.session_state['cap'] = cap
    st.session_state['capturing'] = True

if 'capturing' in st.session_state and st.session_state['capturing']:
    cap = st.session_state['cap']
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels='RGB')

        if stop_capture:
            st.session_state['capturing'] = False
            cap.release()
            st.rerun()
            break
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = base64.b64encode(buffer).decode('utf-8')

        # Decode the image for analysis
        np_image = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        frame = cv2.imdecode(np_image, flags=cv2.IMREAD_COLOR)

        # Analyze emotion
        try:
            result = analyze_emotion(frame)
            if result:
                normalized_result = normalize_scores(result)
                save_data(normalized_result, "output.json")

                # Display results
                emotions = list(normalized_result.keys())
                values = list(normalized_result.values())
                fig, ax = plt.subplots()
                ax.bar(emotions, values, color=['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink'])
                ax.set_ylabel('Intensity (%)')
                ax.set_title('Emotion Analysis')
                chart_placeholder.pyplot(fig)
        except Exception as e:
            st.sidebar.error(f"Error analyzing emotion: {e}")
            break
        
        time.sleep(5)  # Capture every 5 seconds

if stop_capture and 'cap' in st.session_state:
    cap = st.session_state['cap']
    cap.release()
    st.session_state['capturing'] = False
    st.rerun()
