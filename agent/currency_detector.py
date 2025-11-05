import os
import tempfile
from inference_sdk import InferenceHTTPClient
import cv2

class CurrencyDetector:
    """Simple wrapper around Roboflow InferenceHTTPClient to detect currency from frames.

    Usage:
        detector = CurrencyDetector(api_key=os.getenv('ROBOFLOW_API_KEY'))
        result = detector.infer_frame(frame)
        # result -> dict with keys: predictions (list), top_label, top_confidence
    """

    def __init__(self, api_url="https://serverless.roboflow.com", api_key=None, model_id="indian-currency-notes-klhke/2"):
        self.api_url = api_url
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY") 
        # fall back to previous key name used in user's snippet
        if not self.api_key:
            self.client = None
        else:
            self.client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        self.model_id = model_id

    def infer_image_path(self, image_path):
        """Infer using a local image path. Returns the raw result dict from Roboflow or raises an exception."""
        if not self.client:
            raise RuntimeError("Roboflow API key not configured. Set ROBOFLOW_API_KEY in environment or pass api_key to CurrencyDetector.")
        return self.client.infer(image_path, model_id=self.model_id)

    def infer_frame(self, frame):
        """Infer using an OpenCV BGR frame. Writes to a temp file and calls Roboflow.

        Returns a normalized dict: {
            'predictions': [...],
            'top_label': str|None,
            'top_confidence': float|None
        }
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
        try:
            # Ensure frame is BGR (OpenCV) and write
            cv2.imwrite(tmp_path, frame)
            result = self.infer_image_path(tmp_path)
            parsed = {'predictions': []}
            if result and isinstance(result, dict) and 'predictions' in result:
                parsed['predictions'] = result['predictions']
                if len(result['predictions']) > 0:
                    top = result['predictions'][0]
                    parsed['top_label'] = top.get('class')
                    parsed['top_confidence'] = float(top.get('confidence', 0))
                else:
                    parsed['top_label'] = None
                    parsed['top_confidence'] = None
            else:
                parsed['top_label'] = None
                parsed['top_confidence'] = None
            return parsed
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
