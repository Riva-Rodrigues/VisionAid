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
        # smoothing state to reduce flicker between frames
        self.prev_detections = []
        self.smooth_alpha = 0.65  # EMA weight for new detection (0..1). Higher = less smoothing
        self.iou_match_threshold = 0.35
        # depth normalization state: running statistics
        self.depth_min_buffer = []
        self.depth_max_buffer = []
        self.buffer_size = 30  # rolling window of last N frames
        self.setup_models()

    def setup_models(self):
        try:
            # Load YOLOv8 model
            self.yolo_model = YOLO('yolov8n.pt')  
            
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

        # --- Temporal smoothing / simple matching against previous frame ---
        def iou_box(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            xi1 = max(ax1, bx1)
            yi1 = max(ay1, by1)
            xi2 = min(ax2, bx2)
            yi2 = min(ay2, by2)
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            inter = (xi2 - xi1) * (yi2 - yi1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        smoothed = []
        used_prev = set()
        for det in detections:
            best_idx = -1
            best_iou = 0.0
            for idx, p in enumerate(self.prev_detections):
                if idx in used_prev:
                    continue
                if p.get('class') != det.get('class'):
                    continue
                i = iou_box(det['bbox'], p['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_idx = idx

            if best_idx >= 0 and best_iou > self.iou_match_threshold:
                # matched: apply exponential moving average to bbox, center, and confidence
                p = self.prev_detections[best_idx]
                px1, py1, px2, py2 = p['bbox']
                x1, y1, x2, y2 = det['bbox']
                alpha = self.smooth_alpha
                sx1 = int(px1 * (1 - alpha) + x1 * alpha)
                sy1 = int(py1 * (1 - alpha) + y1 * alpha)
                sx2 = int(px2 * (1 - alpha) + x2 * alpha)
                sy2 = int(py2 * (1 - alpha) + y2 * alpha)
                scx = int(p['center'][0] * (1 - alpha) + det['center'][0] * alpha)
                scy = int(p['center'][1] * (1 - alpha) + det['center'][1] * alpha)
                sconf = float(p.get('confidence', 0.0) * (1 - alpha) + det.get('confidence', 0.0) * alpha)
                det['bbox'] = (sx1, sy1, sx2, sy2)
                det['center'] = (scx, scy)
                det['confidence'] = sconf
                used_prev.add(best_idx)

            smoothed.append(det)

        # save current as prev for next frame (shallow copy)
        self.prev_detections = [d.copy() for d in smoothed]
        return smoothed

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
        # guard against invalid bbox ranges
        h, w = depth_map.shape[:2]
        x1c = max(0, min(w - 1, x1))
        x2c = max(0, min(w, x2))
        y1c = max(0, min(h - 1, y1))
        y2c = max(0, min(h, y2))

        if x2c <= x1c or y2c <= y1c:
            # fallback to center pixel depth
            cx = min(w - 1, max(0, (x1 + x2) // 2))
            cy = min(h - 1, max(0, (y1 + y2) // 2))
            median_depth = float(depth_map[cy, cx])
        else:
            roi_depth = depth_map[y1c:y2c, x1c:x2c]
            # reduce edge noise by focusing on central subregion if ROI is large
            rh, rw = roi_depth.shape[:2]
            if rh > 4 and rw > 4:
                cy1 = rh // 4
                cy2 = rh - cy1
                cx1 = rw // 4
                cx2 = rw - cx1
                inner = roi_depth[cy1:cy2, cx1:cx2]
                # median of inner region and full ROI to be robust
                median_depth = float(np.median(inner))
            else:
                median_depth = float(np.median(roi_depth))

        # convert normalized depth to a rough distance (keep previous scaling)
        distance = 3.0 * (1 - median_depth)

        # temporal smoothing with previous detection if available (match by IoU & class)
        try:
            best_idx = -1
            best_iou = 0.0
            for idx, p in enumerate(self.prev_detections):
                if p.get('class') != None:
                    # compute iou with previous bbox
                    px1, py1, px2, py2 = p['bbox']
                    xi1 = max(x1, px1)
                    yi1 = max(y1, py1)
                    xi2 = min(x2, px2)
                    yi2 = min(y2, py2)
                    if xi2 > xi1 and yi2 > yi1:
                        inter = (xi2 - xi1) * (yi2 - yi1)
                        area_a = (x2 - x1) * (y2 - y1)
                        area_b = (px2 - px1) * (py2 - py1)
                        union = area_a + area_b - inter
                        iou = inter / union if union > 0 else 0.0
                    else:
                        iou = 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

            if best_idx >= 0 and best_iou > self.iou_match_threshold and 'distance' in self.prev_detections[best_idx]:
                prev_d = float(self.prev_detections[best_idx]['distance'])
                alpha = self.smooth_alpha
                distance = prev_d * (1 - alpha) + distance * alpha
        except Exception:
            pass

        return max(0.05, float(distance))