import os
import google.generativeai as genai
from dotenv import load_dotenv

class LLMSystem:
    def __init__(self):
        self.setup_llm()

    def setup_llm(self):
        try:
            # Load environment variables from .env file
            load_dotenv()
            genai.configure(api_key=os.getenv("GENAI_API_KEY"))

            # Load the API key from the .env file
            api_key = os.getenv("GENAI_API_KEY")
            if not api_key:
                raise ValueError("API key for Gemini model is not set. Please add GENAI_API_KEY to your .env file.")

            # Log the loaded API Key (for debugging, remove in production)
            print(f"Loaded API Key: {os.getenv('GENAI_API_KEY')}")

            # Initialize the Gemini model
            self.llm_model = genai.GenerativeModel("gemini-1.5-flash")
            print("✓ Gemini model loaded successfully")
        except Exception as e:
            print(f"Error loading Gemini model: {e}")
            self.llm_model = None

    def generate_llm_response(self, user_input, context_info):
        if not self.llm_model:
            print("Gemini model is not loaded.")
            return None
        try:
            # Construct the system context
            prompt = (
                "You are a helpful visual assistant for visually impaired users. "
                f"Current environment: {context_info.get('environment', 'Unknown')}\n"
                f"Recent objects detected: {', '.join(context_info.get('recent_objects', []))}\n"
                f"User's question: {user_input}\n"
                "Provide a helpful, concise response about the visual environment."
            )

            # Generate the response using the Gemini model
            result = self.llm_model.generate_content(prompt)
            response = result.strip()
            print(f"Generated Response: {response}")  # Debugging: Check the generated response

            return response if response and len(response) >= 10 else "Sorry, I couldn't generate a meaningful response."
        except Exception as e:
            print(f"Gemini model error: {e}")
            return "Sorry, an error occurred while generating a response."