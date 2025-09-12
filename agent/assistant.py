import cv2
import threading
import time
from datetime import datetime

from vision import VisionSystem
from llm import LLMSystem
from voice import VoiceSystem
from memory import MemorySystem

class VisualAssistant:
    def __init__(self):
        self.vision = VisionSystem()
        self.llm = LLMSystem()
        self.voice = VoiceSystem()
        self.memory = MemorySystem()
        self.WARNING_DISTANCE = 1.0
        self.FRAME_SKIP = 3
        self.frame_count = 0
        self.is_running = False
        self.listening = False
        self.conversation_history = []
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def process_frame(self, frame):
        detections = self.vision.detect_objects(frame)
        if not detections:
            return frame
        depth_map = self.vision.estimate_depth(frame)
        detections_with_depth = []
        for detection in detections:
            distance = self.vision.calculate_object_distance(depth_map, detection['bbox'])
            detection['distance'] = distance
            detections_with_depth.append(detection)
        self.memory.update_memory(detections_with_depth)
        warnings = self.check_proximity_warnings(detections_with_depth)
        if warnings:
            self.handle_warnings(warnings)
        annotated_frame = self.draw_detections(frame, detections_with_depth)
        return annotated_frame

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

    def check_proximity_warnings(self, detections):
        warnings = []
        for detection in detections:
            if detection['distance'] < self.WARNING_DISTANCE:
                warnings.append({
                    'object': detection['class'],
                    'distance': detection['distance'],
                    'position': detection['position']  # Include position in warnings
                })
        return warnings

    def handle_warnings(self, warnings):
        warning_objects = [
            f"{w['object']} at {w['distance']:.1f} meters ({w['position']})"
            for w in warnings
        ]
        self.memory.context_memory['warnings_given'].append({
            'timestamp': datetime.now(),
            'warnings': warning_objects
        })
        warning_text = f"Warning! Close objects detected: {', '.join(warning_objects)}"
        print(f"⚠️ {warning_text}")
        self.voice.speak(warning_text)

    def listen_for_commands(self):
        command = self.voice.listen()
        if command:
            self.process_voice_command(command)

    def process_voice_command(self, command):
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

    def run(self):
        print("Starting Intelligent Visual Assistant...")
        self.voice.speak("Hello! I'm your intelligent visual assistant. Ask me anything about your surroundings.")
        self.is_running = True
        def voice_listener():
            while self.is_running:
                if not self.listening:
                    self.listen_for_commands()
                time.sleep(0.1)
        voice_thread = threading.Thread(target=voice_listener, daemon=True)
        voice_thread.start()
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                if self.frame_count % self.FRAME_SKIP == 0:
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = frame
                self.frame_count += 1
                cv2.imshow('Intelligent Visual Assistant', processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
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
        print("Intelligent Visual Assistant stopped")