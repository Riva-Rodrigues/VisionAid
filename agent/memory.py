from collections import deque
from datetime import datetime, timedelta

class MemorySystem:
    def __init__(self, memory_duration=300):
        self.memory = deque(maxlen=1000)
        self.scene_summary = {'objects': {}, 'last_updated': datetime.now()}
        self.context_memory = {
            'recent_objects': [],
            'warnings_given': [],
            'user_queries': [],
            'environmental_context': ""
        }
        self.memory_duration = memory_duration

    def update_memory(self, detections_with_depth):
        current_time = datetime.now()
        memory_entry = {'timestamp': current_time, 'detections': detections_with_depth}
        self.memory.append(memory_entry)
        cutoff_time = current_time - timedelta(seconds=self.memory_duration)
        while self.memory and self.memory[0]['timestamp'] < cutoff_time:
            self.memory.popleft()
        self.update_scene_summary(detections_with_depth)
        self.update_context_memory(detections_with_depth)

    def update_scene_summary(self, detections):
        current_objects = {}
        for detection in detections:
            obj_class = detection['class']
            distance = detection['distance']
            if obj_class not in current_objects:
                current_objects[obj_class] = []
            current_objects[obj_class].append({
                'distance': distance,
                'confidence': detection['confidence']
            })
        self.scene_summary = {
            'objects': current_objects,
            'last_updated': datetime.now()
        }

    def update_context_memory(self, detections):
        current_objects = [det['class'] for det in detections]
        self.context_memory['recent_objects'] = list(set(
            self.context_memory['recent_objects'][-20:] + current_objects
        ))
        if detections:
            close_objects = [det for det in detections if det['distance'] < 2.0]
            far_objects = [det for det in detections if det['distance'] >= 2.0]
            context_parts = []
            if close_objects:
                close_items = [obj['class'] for obj in close_objects]
                context_parts.append(f"Nearby objects: {', '.join(set(close_items))}")
            if far_objects:
                far_items = [obj['class'] for obj in far_objects]
                context_parts.append(f"Distant objects: {', '.join(set(far_items))}")
            self.context_memory['environmental_context'] = ". ".join(context_parts)

    def get_recent_detections(self, seconds=10):
        if not self.memory:
            return []
        recent_time = datetime.now() - timedelta(seconds=seconds)
        recent_detections = []
        for entry in reversed(self.memory):
            if entry['timestamp'] >= recent_time:
                recent_detections.extend(entry['detections'])
            else:
                break
        return recent_detections