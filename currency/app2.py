from flask import Flask, jsonify
from inference_sdk import InferenceHTTPClient
import cv2, os, tempfile, time, pyttsx3

app = Flask(__name__)

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="0WPKLUbiK37fd3FFofr6"
)

# Global detection result
last_result = {"message": "Waiting for detection..."}

def speak(text):
    """Re-initialize engine each time to avoid run loop issues"""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def detect_currency():
    global last_result

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        last_result = {"error": "Cannot access webcam"}
        return

    last_capture = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            last_result = {"error": "Failed to capture frame"}
            break

        # Show live webcam window
        cv2.imshow("Currency Detection", frame)

        # Every 10 seconds, run detection
        if time.time() - last_capture >= 10:
            last_capture = time.time()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp_path = temp.name
                cv2.imwrite(temp_path, frame)

            try:
                result = CLIENT.infer(temp_path, model_id="indian-currency-notes-klhke/2")
                last_result = result

                if "predictions" in result and len(result["predictions"]) > 0:
                    label = result["predictions"][0]["class"]
                    print(f"Predicted: {label} rupee note")
                    speak(f"This is {label} rupee note")
                else:
                    print("No currency detected")
                    speak("No currency detected")

            except Exception as e:
                last_result = {"error": str(e)}
                print("Error:", e)

            finally:
                os.remove(temp_path)

        # Press 'q' to quit webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def index():
    return jsonify(last_result)

if __name__ == "__main__":
    detect_currency()
