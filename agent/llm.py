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
            self.llm_model = genai.GenerativeModel("gemini-2.5-flash")
            print("✓ Gemini model loaded successfully")
        except Exception as e:
            print(f"Error loading Gemini model: {e}")
            self.llm_model = None

    def generate_llm_response(self, user_input, context_info):
        if not self.llm_model:
            print("Gemini model is not loaded.")
            return None
        try:
            # Process detailed object information
            detailed_objects = []
            for obj in context_info.get('recent_detections', []):
                details = []
                # Add position information
                if obj.get('position'):
                    details.append(f"located {obj['position']}")
                # Add distance information
                if obj.get('distance'):
                    details.append(f"about {obj['distance']:.1f} meters away")
                # Add movement information
                if obj.get('movement'):
                    details.append(f"is {obj['movement']}")
                
                obj_desc = f"a {obj['class']} ({', '.join(details)})"
                detailed_objects.append(obj_desc)

            # Construct the enhanced system context
            prompt = (
                "You are a helpful visual assistant for visually impaired users. Use the following detailed information about "
                "the environment to provide specific, location-aware responses. When asked about distances or locations, "
                "include the exact positioning and measurements provided.\n\n"
                f"Current environment: {context_info.get('environment', 'Unknown')}\n"
                "Detailed scene description:\n"
                f"{'; '.join(detailed_objects)}\n"
                f"User's question: {user_input}\n\n"
                "Provide a helpful, natural response that incorporates the specific position, distance, and movement information when relevant."
            )

            # Generate the response using the Gemini model
            result = self.llm_model.generate_content(prompt)
            # print(f"Raw Response: {result}")  # Debugging: Check the raw response

            # Extract the content from the result object
            if result and result.candidates:
                response = result.candidates[0].content.parts[0].text  # Access attributes directly
                response = response.strip()  # Ensure no leading/trailing whitespace
                # print(f"Generated Response: {response}")  # Debugging: Check the generated response
                return response if len(response) >= 10 else "Sorry, I couldn't generate a meaningful response."
            else:
                print("No candidates found in the response.")
                return "Sorry, I couldn't generate a meaningful response."
        except Exception as e:
            print(f"Gemini model error: {e}")
            return "Sorry, an error occurred while generating a response."