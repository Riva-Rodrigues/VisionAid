import cv2
import threading
import time
from datetime import datetime

from vision import VisionSystem
from navigation_vision import NavigationVisionSystem
from llm import LLMSystem
from voice import VoiceSystem
from memory import MemorySystem

# Set of navigation classes considered hazardous for the navigation model
NAV_HAZARDOUS_CLASSES = {
    'bicycle', 'bus', 'car', 'crosswalk', 'elevator', 'escalator', 'green_light',
    'motorcycle', 'person', 'pole', 'red_light', 'reflective_cone', 'sign',
    'stairs', 'trash_can', 'tree', 'truck', 'warning_column'
}

COCO_HAZARDOUS_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "suitcase", "tie", "skateboard", "surfboard", "sports ball", "kite",
    "potted plant", "traffic cone"
}

class VisualAssistant:
    def __init__(self):
        self.vision = VisionSystem()
        self.navigation_vision = NavigationVisionSystem()
        self.llm = LLMSystem()
        self.voice = VoiceSystem()
        self.memory = MemorySystem()
        self.WARNING_DISTANCE = 1.0
        self.FRAME_SKIP = 4
        self.frame_count = 0
        self.is_running = False
        self.listening = False
        self.conversation_history = []
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.latest_processed = None
        self.processed_lock = threading.Lock()
        self.use_navigation_model = False

    def process_frame(self, frame):
        if self.use_navigation_model:
            detections = self.navigation_vision.detect_objects(frame)
            if not detections:
                return frame, []
            depth_map = self.navigation_vision.estimate_depth(frame)
            detections_with_depth = []
            for detection in detections:
                distance = self.navigation_vision.calculate_object_distance(depth_map, detection['bbox'])
                detection['distance'] = distance
                detections_with_depth.append(detection)
            warnings = self.check_proximity_warnings(detections_with_depth, navigation_mode=True)
        else:
            detections = self.vision.detect_objects(frame)
            if not detections:
                return frame, []
            depth_map = self.vision.estimate_depth(frame)
            detections_with_depth = []
            for detection in detections:
                distance = self.vision.calculate_object_distance(depth_map, detection['bbox'])
                detection['distance'] = distance
                detections_with_depth.append(detection)
            warnings = self.check_proximity_warnings(detections_with_depth, navigation_mode=False)
        self.memory.update_memory(detections_with_depth)
        annotated_frame = self.draw_detections(frame.copy(), detections_with_depth)
        return annotated_frame, warnings

    def draw_detections(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            distance = detection['distance']
            confidence = detection['confidence']
            color = (0, 0, 255) if distance < self.WARNING_DISTANCE else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {distance:.1f}m ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def check_proximity_warnings(self, detections, navigation_mode=False):
        warnings = []
        hazard_set = NAV_HAZARDOUS_CLASSES if navigation_mode else COCO_HAZARDOUS_CLASSES
        for detection in detections:
            if (detection['distance'] < self.WARNING_DISTANCE and
                detection['class'] in hazard_set):
                warnings.append({
                    'object': detection['class'],
                    'distance': detection['distance'],
                    'position': detection['position']
                })
        return warnings

    def handle_warnings(self, warnings):
        warning_objects = [
            f"{w['object']} that is {w.get('movement','stationary')} at {w['distance']:.1f} meters ({w['position']})"
            for w in warnings
        ]
        self.memory.context_memory['warnings_given'].append({
            'timestamp': datetime.now(),
            'warnings': warning_objects
        })
        warning_text = f"Warning! Close objects detected: {', '.join(warning_objects)}"
        print(f"⚠️ {warning_text}")
        self.voice.speak(warning_text)
        time.sleep(0.5)

    def listen_for_commands(self):
        try:
            command = self.voice.listen()
            if command:
                self.process_voice_command(command)
        except Exception as e:
            print(f"[Voice Command Error] {e}")
        finally:
            self.listening = False

    def process_voice_command(self, command):
        try:
            # Navigation mode switching logic
            if any(kw in command for kw in ["navigation", "navigate", "guide me"]):
                if not self.use_navigation_model:
                    self.use_navigation_model = True
                    self.voice.speak("Navigation mode activated. I will now guide you with navigation-specific information.")
                else:
                    self.voice.speak("Navigation mode is already activated.")
                return
            if any(kw in command for kw in ["stop navigation", "exit navigation", "normal mode", "object detection"]):
                if self.use_navigation_model:
                    self.use_navigation_model = False
                    self.voice.speak("Navigation mode deactivated. Returning to normal object detection.")
                else:
                    self.voice.speak("Navigation mode is already deactivated.")
                return
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user': command
            })
            context_info = {
                'environment': self.memory.context_memory['environmental_context'],
                'recent_objects': self.memory.context_memory['recent_objects'],
                'close_objects': [
                    f"{obj['class']} ({obj['position']})"
                    for obj in self.memory.get_recent_detections()
                    if obj['distance'] < 2.0
                ],
                'warnings': self.memory.context_memory['warnings_given'][-3:]
            }
            if any(word in command for word in ['stop', 'quit', 'exit', 'goodbye']):
                self.is_running = False
                response = "Goodbye! Visual assistant stopping."
                self.voice.speak(response)
                return
            response = self.llm.generate_llm_response(command, context_info)
            if not response:
                response = "I can help you understand your surroundings. Ask me what's in front of you, if it's safe to walk, or about specific objects you're looking for."
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'assistant': response
            })
            print(f"Response: {response}")
            self.voice.speak(response)
            time.sleep(0.5)
        finally:
            self.listening = False

    def run(self):
        print("Starting Intelligent Visual Assistant...")
        self.voice.speak("Hello! I'm your intelligent visual assistant. Ask me anything about your surroundings.")
        self.is_running = True

        def voice_listener():
            while self.is_running:
                try:
                    if not self.listening:
                        self.listening = True
                        self.listen_for_commands()
                    time.sleep(0.1)
                except Exception as e:
                    print(f"[Voice Listener Thread Error] {e}")
                    self.listening = False
                    time.sleep(1)

        def processing_worker():
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                if self.frame_count % self.FRAME_SKIP == 0:
                    processed, warnings = self.process_frame(frame)
                    with self.processed_lock:
                        self.latest_processed = processed
                    if warnings:
                        self.handle_warnings(warnings)
                else:
                    with self.processed_lock:
                        self.latest_processed = frame.copy()
                self.frame_count += 1
                time.sleep(0.01)

        voice_thread = threading.Thread(target=voice_listener, daemon=True)
        process_thread = threading.Thread(target=processing_worker, daemon=True)
        voice_thread.start()
        process_thread.start()
        try:
            while self.is_running:
                with self.processed_lock:
                    show_frame = self.latest_processed
                if show_frame is not None:
                    cv2.imshow('Intelligent Visual Assistant', show_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping application...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.is_running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("Intelligent Visual Assistant stopped")