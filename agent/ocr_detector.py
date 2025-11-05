import cv2
import base64
import os
from mistralai import Mistral

class OCRDetector:
    """Wrapper around Mistral's OCR API to extract text from frames.

    Usage:
        detector = OCRDetector(api_key=os.getenv('MISTRAL_API_KEY'))
        text = detector.process_frame(frame)
        # text -> str with extracted text from the frame
    """

    def __init__(self, api_key=None, model="mistral-ocr-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            print("Warning: Mistral API key not configured. Set MISTRAL_API_KEY in environment.")
            self.client = None
        else:
            try:
                from mistralai import Mistral
                self.client = Mistral(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Mistral client: {e}")
                self.client = None
        self.model = model

    def frame_to_base64(self, frame):
        """Convert an OpenCV frame to base64 string."""
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def process_frame(self, frame):
        """Process a frame through Mistral OCR and return extracted text.
        
        Args:
            frame: OpenCV BGR frame

        Returns:
            str: Extracted text from the frame, or error message if processing failed
        """
        if not self.client:
            return "OCR is not configured. Set MISTRAL_API_KEY in environment and restart."

        try:
            base64_frame = self.frame_to_base64(frame)
            response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_frame}"
                },
                include_image_base64=False
            )

            # Extract text from all pages
            text_parts = []
            for page in response.pages:
                if page.markdown.strip():
                    text_parts.append(page.markdown.strip())

            if not text_parts:
                return "No text detected in the image."
            
            return "\n".join(text_parts)

        except Exception as e:
            print(f"OCR processing error: {e}")
            return f"Failed to process image: {str(e)}"