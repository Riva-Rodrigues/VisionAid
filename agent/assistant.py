import cv2
import threading
import time
from datetime import datetime

from vision import VisionSystem
from navigation_vision import NavigationVisionSystem
from llm import LLMSystem
from voice import VoiceSystem
from memory import MemorySystem

NAVIGATION_KEYWORDS = [
    "navigate", "street", "road", "cross", "traffic", "walk", "direction", "intersection", "signal", "vehicle"
]
GENERAL_KEYWORDS = [
    "general", "object", "identify", "what is", "describe", "see", "around", "furniture", "chair", "table"
]

NAV_HAZARD_CLASSES = {
    "car", "bus", "truck", "motorcycle", "bicycle",
    "red_light", "green_light", "crosswalk",
    "stairs", "elevator", "escalator", "pole", "sign",
    "reflective_cone", "warning_column", "trash_can", "tree", "person"
}
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
TRAFFIC_SIGNALS = {"red_light", "green_light"}
MOBILITY_AIDS = {"stairs", "elevator", "escalator"}

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
        self.mode = "general"  # "navigation" or "general"

    def auto_switch_mode(self, command):
        nav_trigger = any(word in command for word in NAVIGATION_KEYWORDS)
        gen_trigger = any(word in command for word in GENERAL_KEYWORDS)
        prev_mode = self.mode
        if nav_trigger and self.mode != "navigation":
            self.mode = "navigation"
            self.voice.speak("Navigation mode activated.")
            print("Switched to Navigation Mode")
        elif gen_trigger and self.mode != "general":
            self.mode = "general"
            self.voice.speak("General mode activated.")
            print("Switched to General Mode")

    def process_frame(self, frame):
        if self.mode == "navigation":
            detections = self.navigation_vision.detect_objects(frame)
            if not detections:
                return frame, []
            depth_map = self.navigation_vision.estimate_depth(frame)
            detections_with_depth = []
            for detection in detections:
                distance = self.navigation_vision.calculate_object_distance(depth_map, detection['bbox'])
                detection['distance'] = distance
                detections_with_depth.append(detection)
            warnings = self.navigation_warnings(detections_with_depth)
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
            warnings = self.general_warnings(detections_with_depth)
        self.memory.update_memory(detections_with_depth)
        annotated_frame = self.draw_detections(frame.copy(), detections_with_depth)
        return annotated_frame, warnings

    def draw_detections(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            distance = detection.get('distance', 0)
            confidence = detection.get('confidence', 0)
            movement = detection.get('movement')
            color = (0, 0, 255) if distance < self.WARNING_DISTANCE else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {distance:.1f}m ({confidence:.2f})"
            if movement:
                label += f" [{movement}]"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def navigation_warnings(self, detections):
        warnings = []
        for d in detections:
            cls = d['class']
            dist = d['distance']
            movement = d.get('movement', 'stationary')
            # Critical vehicle warnings at 3+ meters
            if cls in VEHICLES and dist < 3.0:
                warnings.append({**d, 'reason': 'approaching vehicle'})
            # Traffic signals at 5+ meters
            elif cls in TRAFFIC_SIGNALS and dist < 5.0:
                warnings.append({**d, 'reason': 'traffic signal'})
            # Mobility aids at 2+ meters
            elif cls in MOBILITY_AIDS and dist < 2.0:
                warnings.append({**d, 'reason': 'mobility aid'})
            # Other hazards at close range
            elif cls in NAV_HAZARD_CLASSES and dist < self.WARNING_DISTANCE:
                warnings.append({**d, 'reason': 'obstacle'})
        return warnings

    def general_warnings(self, detections):
        # Only warn for close hazardous objects (using your COCO list)
        warnings = []
        hazard_set = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "traffic light", "fire hydrant", "stop sign", "parking meter"}
        for d in detections:
            if d['class'] in hazard_set and d['distance'] < self.WARNING_DISTANCE:
                warnings.append(d)
        return warnings

    def handle_warnings(self, warnings):
        if not warnings:
            return
        if self.mode == "navigation":
            warning_objects = [
                f"{w['class']} that is {w.get('movement','stationary')} at {w['distance']:.1f} meters ({w['position']}) [{w.get('reason','hazard')}]"
                for w in warnings
            ]
        else:
            warning_objects = [
                f"{w['class']} at {w['distance']:.1f} meters ({w['position']})"
                for w in warnings
            ]
        self.memory.context_memory['warnings_given'].append({
            'timestamp': datetime.now(),
            'warnings': warning_objects
        })
        warning_text = f"Warning! {', '.join(warning_objects)}"
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
        command = command.lower()
        self.auto_switch_mode(command)
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user': command
        })

        # --- Context-Aware Navigation Handling ---
        if self.mode == "navigation":
            if "safe to cross" in command or "can i cross" in command:
                latest = self.memory.get_recent_detections()
                vehicles = [o for o in latest if o['class'] in VEHICLES and o['distance'] < 3.0]
                red_lights = [o for o in latest if o['class'] == "red_light" and o['distance'] < 5.0]
                green_lights = [o for o in latest if o['class'] == "green_light" and o['distance'] < 5.0]
                crosswalks = [o for o in latest if o['class'] == "crosswalk"]
                if vehicles:
                    response = "It is NOT safe to cross. There are vehicles nearby."
                elif red_lights:
                    response = "It is NOT safe to cross. The traffic light is red."
                elif green_lights and crosswalks:
                    response = "It is safe to cross. The light is green and a crosswalk is detected."
                elif crosswalks:
                    response = "A crosswalk is detected ahead. Please check for traffic before crossing."
                else:
                    response = "I cannot determine if it is safe to cross. Please be cautious."
                self.voice.speak(response)
                print(response)
                return
            elif "what vehicles" in command or "which vehicles" in command:
                latest = self.memory.get_recent_detections()
                vehicles = [o for o in latest if o['class'] in VEHICLES]
                if not vehicles:
                    response = "No vehicles detected nearby."
                else:
                    desc = [f"{v['class']} at {v['distance']:.1f} meters {v['position']}" for v in vehicles]
                    response = "Vehicles detected: " + ", ".join(desc)
                self.voice.speak(response)
                print(response)
                return
            elif "stairs" in command or "elevator" in command or "escalator" in command:
                latest = self.memory.get_recent_detections()
                aids = [o for o in latest if o['class'] in MOBILITY_AIDS]
                if not aids:
                    response = "No stairs, elevator, or escalator detected nearby."
                else:
                    desc = [f"{a['class']} at {a['distance']:.1f} meters {a['position']}" for a in aids]
                    response = "Mobility aids detected: " + ", ".join(desc)
                self.voice.speak(response)
                print(response)
                return

        # --- Context-Aware General Handling ---
        if self.mode == "general":
            if "what objects" in command or "what is around" in command or "describe" in command or "see" in command:
                latest = self.memory.get_recent_detections()
                if not latest:
                    response = "I do not see any objects right now."
                else:
                    desc = [
                        f"{o['class']} at {o['distance']:.1f} meters {o['position']}"
                        for o in latest if o['distance'] < 5.0
                    ]
                    response = "Objects detected: " + ", ".join(desc)
                self.voice.speak(response)
                print(response)
                return
            elif "chair" in command or "table" in command or "furniture" in command:
                latest = self.memory.get_recent_detections()
                furniture = [o for o in latest if o['class'] in {"chair", "table", "couch", "sofa"}]
                if not furniture:
                    response = "No furniture detected nearby."
                else:
                    desc = [f"{f['class']} at {f['distance']:.1f} meters {f['position']}" for f in furniture]
                    response = "Furniture detected: " + ", ".join(desc)
                self.voice.speak(response)
                print(response)
                return

        # --- Fallback to LLM ---
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