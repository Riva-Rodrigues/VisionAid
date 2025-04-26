import streamlit as st

import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue
import pyttsx3
import warnings
import os

st.set_option('client.showErrorDetails', False)
# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Page configuration
st.set_page_config(
    page_title="Visual Assistance Platform",
    page_icon="👁️",
    layout="wide"
)

# Initialize text-to-speech engine
def create_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    return engine

# Queue for voice commands
voice_queue = queue.Queue()

def speak_text(text):
    """Function to convert text to speech"""
    try:
        # Create a new engine instance each time to avoid issues
        engine = create_tts_engine()
        engine.say(text)
        engine.runAndWait()
        # Print for debugging
        print(f"Speaking: {text}")
    except Exception as e:
        print(f"Speech error: {e}")

def voice_worker():
    """Worker thread to process voice commands"""
    while True:
        text = voice_queue.get()
        if text is None:
            break
        speak_text(text)
        voice_queue.task_done()

# Start voice worker thread
voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor with YOLO model"""
        self.model = None
        self.last_announcement_time = 0
        self.announcement_cooldown = 1.5  # seconds between announcements
        self.detected_objects = set()  # Simplified to just track current objects
        self.previous_objects = set()  # Track previous frame's objects
        
        # Color mapping for bounding boxes (for consistent colors per class)
        self.colors = {}

    def load_model(self, model_path="yolov8n.pt"):
        """Load the YOLO model"""
        if self.model is None:
            try:
                self.model = YOLO(model_path)
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return True
    
    def get_color(self, class_id):
        """Get a consistent color for a class ID"""
        if class_id not in self.colors:
            # Generate a random color
            self.colors[class_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        return self.colors[class_id]

    def process_frame(self, frame, confidence_threshold=0.5, announce=False):
        """Process a single frame for object detection"""
        if self.model is None:
            return frame, []

        # Make a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Detect objects
        results = self.model(frame, conf=confidence_threshold)[0]
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else []
        confs = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else []
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) > 0 else []
        
        detections = list(zip(boxes, confs, class_ids))
        
        # Collect current objects
        self.detected_objects = set()
        for _, _, class_id in detections:
            obj_name = self.model.names[class_id]
            self.detected_objects.add(obj_name)
        
        # Check for objects to announce - simplified approach
        current_time = time.time()
        if announce and current_time - self.last_announcement_time > self.announcement_cooldown:
            # Find new objects that weren't in the previous frame
            new_objects = self.detected_objects - self.previous_objects
            
            if new_objects:
                # Announce new objects
                if len(new_objects) == 1:
                    announcement = f"I see a {list(new_objects)[0]}"
                else:
                    announcement = f"I see: {', '.join(new_objects)}"
                
                # Direct speak for immediate feedback and also queue it
                speak_text(announcement)
                self.last_announcement_time = current_time
        
        # Update previous objects for next comparison
        self.previous_objects = self.detected_objects.copy()
        
        # Draw bounding boxes and labels
        for box, conf, class_id in detections:
            # Get label and color
            label = f"{self.model.names[class_id]} {conf:.2f}"
            color = self.get_color(class_id)
            
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_size[1] - 10), 
                (x1 + text_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )

        return annotated_frame, detections

# Main app
def main():
    st.title("Visual Assistance Platform")
    st.markdown("### Object Detection Module")
    
    processor = VideoProcessor()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        model_path = st.selectbox(
            "Select YOLO model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            index=0
        )
        
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        voice_feedback = st.checkbox("Enable Voice Feedback", value=True)
        
        # Test voice button for debugging
        if st.button("Test Voice"):
            speak_text("This is a voice test. If you can hear this, voice is working.")
        
        st.divider()
        st.markdown("### About")
        st.info(
            "This application uses YOLOv8 for real-time object detection to assist "
            "visually impaired individuals in identifying objects around them."
        )

    # Load model
    with st.spinner("Loading YOLO model..."):
        model_loaded = processor.load_model(model_path)
    
    if model_loaded:
        st.success("Model loaded successfully!")
        # Speak after model is loaded
        if voice_feedback:
            speak_text("Object detection is ready. I will announce objects as I see them.")
    else:
        st.error("Failed to load model. Please check your connection and try again.")
        return

    # Webcam feed
    st.header("Live Camera Feed")
    
    # Create a placeholder for webcam feed
    video_placeholder = st.empty()
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        st.error("Error: Could not access webcam. Please check your camera connection.")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Status display
    status_container = st.container()
    status_text = status_container.empty()
    
    # Create a button to stop the app
    stop_button = st.button("Stop Detection")
    frame_counter = 0
    
    # Capture and process webcam feed
    try:
        while not stop_button:
            ret, frame = cap.read()
            frame_counter += 1
            
            if not ret:
                st.error("Error: Failed to receive frame from webcam.")
                break
            
            # Convert BGR to RGB for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Only process every other frame to reduce CPU load and allow TTS time to work
            if frame_counter % 2 == 0:
                # Process frame for object detection
                processed_frame, detections = processor.process_frame(
                    frame_rgb, 
                    confidence_threshold=confidence,
                    announce=voice_feedback
                )
                
                # Display the processed frame
                video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Update status text
                if len(detections) > 0:
                    detected_items = []
                    for _, _, class_id in detections:
                        obj_name = processor.model.names[class_id]
                        detected_items.append(obj_name)
                        
                    status_text.markdown(f"**Detected objects:** {', '.join(set(detected_items))}")
                else:
                    status_text.markdown("**No objects detected**")
            
            # Add a delay to reduce CPU usage and give voice time to process
            time.sleep(0.2)
            
            # Need to rerun to check button state
            if frame_counter % 10 == 0:  # Check less frequently
                stop_button = st.button("Stop Detection", key=f"stop_{frame_counter}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Release webcam when done or if an error occurs
        cap.release()
        # Signal voice thread to terminate
        voice_queue.put(None)
        st.success("Detection stopped. You can close the browser tab or refresh to restart.")

if __name__ == "__main__":
    main()