import torch
import cv2
import numpy as np
from ultralytics import YOLO

def non_max_suppression(detections, iou_threshold=0.5):
    """Remove duplicate detections based on bounding box overlap (IOU)"""
    if not detections:
        return []
    
    # Sort by confidence in descending order
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    filtered = []
    
    for detection in sorted_dets:
        is_duplicate = False
        x1, y1, x2, y2 = detection['bbox']
        area1 = (x2 - x1) * (y2 - y1)
        
        for kept in filtered:
            kx1, ky1, kx2, ky2 = kept['bbox']
            area2 = (kx2 - kx1) * (ky2 - ky1)
            
            # Calculate intersection
            xi1 = max(x1, kx1)
            yi1 = max(y1, ky1)
            xi2 = min(x2, kx2)
            yi2 = min(y2, ky2)
            
            if xi2 > xi1 and yi2 > yi1:
                intersection = (xi2 - xi1) * (yi2 - yi1)
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                # If same class and high IOU, it's a duplicate
                if detection['class'] == kept['class'] and iou > iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append(detection)
    
    return filtered

class VisionSystem:
    def __init__(self):
        self.setup_models()

    def setup_models(self):
        try:
            # Load YOLOv8 model
            self.yolo_model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
            
            # Load MiDaS depth estimation model
            self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.depth_model.eval()
            self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            print("✓ Vision models loaded successfully")
        except Exception as e:
            print(f"Error loading vision models: {e}")

    def detect_objects(self, frame):
        # Run YOLOv8 inference
        results = self.yolo_model(frame, verbose=False)
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        # Process results from YOLOv8
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates, confidence, and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        class_name = self.yolo_model.names[cls]
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                        # Determine the relative position of the object
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

                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'position': "-".join(position)  # e.g., "top-left"
                        })
        
        # Apply NMS to remove duplicate detections
        detections = non_max_suppression(detections, iou_threshold=0.5)
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
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        return depth_map

    def calculate_object_distance(self, depth_map, bbox):
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]
        median_depth = np.median(roi_depth)
        distance = 3.0 * (1 - median_depth)
        return max(0.1, distance)