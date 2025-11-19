import torch
import numpy as np
from ultralytics import YOLO
import time

NAV_CLASSES = [
    'bicycle', 'bus', 'car', 'crosswalk', 'elevator', 'escalator', 'green_light',
    'motorcycle', 'person', 'pole', 'red_light', 'reflective_cone', 'sign',
    'stairs', 'trash_can', 'tree', 'truck', 'warning_column'
]

class NavigationVisionSystem:
    def __init__(self):
        # Load YOLO navigation model
        self.yolo_model = YOLO('nav.pt')
        self.class_names = NAV_CLASSES
        # Depth model
        self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.depth_model.eval()
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        # Movement history
        self.object_history = {}  # { (class, pos_bucket): {center, timestamp} }
        self.movement_threshold = 20  # pixels; tune as needed
        self.time_threshold = 1.0     # seconds
        # Depth normalization state: running statistics
        self.depth_min_buffer = []
        self.depth_max_buffer = []
        self.buffer_size = 30  # rolling window of last N frames

    def _get_pos_bucket(self, center):
        # Bucketing helps to match objects even with slight detection jitter
        return (round(center[0]/30), round(center[1]/30))  # buckets of 30px

    def detect_objects(self, frame):
        results = self.yolo_model(frame, verbose=False)
        detections = []
        frame_height, frame_width = frame.shape[:2]
        now = time.time()
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        class_name = self.class_names[cls]
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        position = []
                        if center_y < frame_height / 3:
                            position.append("top")
                        elif center_y > 2 * frame_height / 3:
                            position.append("bottom")
                        if center_x < frame_width / 3:
                            position.append("left")
                        elif center_x > 2 * frame_width / 3:
                            position.append("right")
                        if not position:
                            position.append("center")
                        # Movement estimation
                        pos_bucket = self._get_pos_bucket((center_x, center_y))
                        key = (class_name, pos_bucket)
                        movement = "stationary"
                        if key in self.object_history:
                            prev = self.object_history[key]
                            dist = ((center_x - prev['center'][0])**2 + (center_y - prev['center'][1])**2)**0.5
                            dt = now - prev['timestamp']
                            if dt < self.time_threshold and dist > self.movement_threshold:
                                movement = "moving"
                        # Update history
                        self.object_history[key] = {'center': (center_x, center_y), 'timestamp': now}
                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'position': "-".join(position),
                            'movement': movement
                        })
        return detections

    def estimate_depth(self, frame):
        input_batch = self.transform(frame)
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        
        # Normalize per-frame first
        frame_min = float(depth_map.min())
        frame_max = float(depth_map.max())
        
        # Update running statistics
        self.depth_min_buffer.append(frame_min)
        self.depth_max_buffer.append(frame_max)
        if len(self.depth_min_buffer) > self.buffer_size:
            self.depth_min_buffer.pop(0)
        if len(self.depth_max_buffer) > self.buffer_size:
            self.depth_max_buffer.pop(0)
        
        # Use median of recent min/max for stable normalization
        stable_min = float(np.median(self.depth_min_buffer))
        stable_max = float(np.median(self.depth_max_buffer))
        
        # Normalize using stable statistics
        if stable_max > stable_min:
            depth_map = (depth_map - stable_min) / (stable_max - stable_min)
        else:
            depth_map = (depth_map - frame_min) / (frame_max - frame_min)
        
        # Clamp to [0, 1]
        depth_map = np.clip(depth_map, 0.0, 1.0)
        return depth_map

    def calculate_object_distance(self, depth_map, bbox):
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]
        median_depth = np.median(roi_depth)
        distance = 3.0 * (1 - median_depth)
        return max(0.1, distance)