import streamlit as st
import google.generativeai as genai
import datetime
import time
import re
import os
import asyncio
from pathlib import Path
import logging
from googletrans import Translator, LANGUAGES
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
import tempfile
import nest_asyncio
from typing import List, Optional, Any
from typing import Dict, Optional, Tuple
import threading
import wave
import pyaudio

# Enable nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Translation configuration
GTTS_LANG_CODES = {
    'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'pt': 'pt',
    'ru': 'ru', 'hi': 'hi', 'ja': 'ja', 'ko': 'ko', 'zh-cn': 'zh-CN',
    'ar': 'ar', 'bn': 'bn', 'nl': 'nl', 'el': 'el', 'gu': 'gu', 'hu': 'hu',
    'id': 'id', 'kn': 'kn', 'ml': 'ml', 'mr': 'mr', 'ne': 'ne', 'pl': 'pl',
    'ta': 'ta', 'te': 'te', 'th': 'th', 'tr': 'tr', 'ur': 'ur', 'vi': 'vi'
}

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.stream = None
        self.audio = None
        self.thread = None

    def start_recording(self):
        self.recording = True
        self.audio_frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        return self.save_recording()

    def _record(self):
        while self.recording:
            data = self.stream.read(1024)
            self.audio_frames.append(data)

    def save_recording(self):
        if not self.audio_frames:
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wf = wave.open(tmp_file.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            return tmp_file.name


class GeminiContextCache:
    def __init__(self):
        self.cached_models = {}
        self.MIN_TOKENS = 32768
        self.general_model = None
        self.translator = Translator()
        
        # Initialize API key from environment variable
        api_key ="AIzaSyCIGIFXMaYtlHanVHraamT8hFZH4RBL-_E"
        if api_key:
            self.setup_api(api_key)
        else:
            logger.error("GOOGLE_API_KEY not found in environment variables")

    def setup_api(self, api_key: str) -> bool:
        try:
            genai.configure(api_key=api_key)
            self.general_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
            return True
        except Exception as e:
            logger.error(f"Invalid API key: {str(e)}")
            return False

    def process_file(self, uploaded_file):
        try:
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                try:
                    file_data = genai.upload_file(path=str(temp_path))
                    while file_data.state.name == "PROCESSING":
                        st.write("Processing...")
                        time.sleep(2)
                        file_data = genai.get_file(file_data.name)
                    status.update(label="Processing complete!", state="complete")
                except Exception as e:
                    status.update(label=f"Error: {str(e)}", state="error")
                    return None

            temp_path.unlink()
            return file_data
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

    def create_padding_file(self) -> Optional[Any]:
        try:
            temp_path = Path("temp_padding.txt")
            padding_text = "This is padding content. " * 8000
            
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(padding_text)

            padding_data = genai.upload_file(path=str(temp_path))
            while getattr(padding_data, 'state', None) and padding_data.state.name == "PROCESSING":
                time.sleep(1)
                padding_data = genai.get_file(padding_data.name)
            
            temp_path.unlink()
            return padding_data
        except Exception as e:
            st.error(f"Error creating padding: {str(e)}")
            return None

    def update_cache(self, new_files: List, cache_name: str, 
                    system_instruction: str, ttl_minutes: int) -> Optional[str]:
        try:
            processed_files = []
            for uploaded_file in new_files:
                file_data = self.process_file(uploaded_file)
                if file_data:
                    processed_files.append(file_data)

            if not processed_files:
                st.error("No files were successfully processed.")
                return None

            existing_cache = self.cached_models.get(cache_name)
            if existing_cache:
                all_files = existing_cache.cached_content.contents + processed_files
            else:
                all_files = processed_files

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    cache = genai.caching.CachedContent.create(
                        model="models/gemini-1.5-flash-001",
                        display_name=cache_name,
                        system_instruction=system_instruction,
                        contents=all_files,
                        ttl=datetime.timedelta(minutes=ttl_minutes)
                    )
                    
                    self.cached_models[cache_name] = genai.GenerativeModel.from_cached_content(
                        cached_content=cache
                    )
                    return cache_name
                    
                except Exception as e:
                    if "Cached content is too small" in str(e) and attempt < max_retries - 1:
                        st.warning("Adding padding content...")
                        padding_data = self.create_padding_file()
                        if padding_data:
                            all_files.append(padding_data)
                        continue
                    else:
                        st.error(f"Error updating cache: {str(e)}")
                        return None
        except Exception as e:
            st.error(f"Error in update_cache: {str(e)}")
            return None

    def query_model(self, prompt: str, chat_history: list, use_files: bool = False) -> dict:
        try:
            if prompt.strip().lower().startswith('translate'):
                return self.handle_translation(prompt)
                
            gemini_history = []
            for msg in chat_history[:-1]:
                if msg["role"] == "user":
                    gemini_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    gemini_history.append({"role": "model", "parts": [{"text": msg["content"]}]})
            
            if use_files and st.session_state.current_cache in self.cached_models:
                model = self.cached_models[st.session_state.current_cache]
                prompt_with_context = f"""Using the context from the uploaded files ({st.session_state.current_cache}), answer:

{prompt}

Reference specific file contents where applicable. If unsure, state that information isn't available."""
            else:
                model = self.general_model
                prompt_with_context = prompt

            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(prompt_with_context)
            
            return {'text': response.text, 'audio_path': None}

        except Exception as e:
            error_msg = f"Error getting response: {str(e)}"
            st.error(error_msg)
            return {'text': error_msg, 'audio_path': None}
        
    def run_async(self, coroutine):
        """Helper method to run async code in sync context"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)

    async def async_translate(self, text, dest_lang):
        """Async method to handle translation"""
        detected = await self.translator.detect(text)
        translated = await self.translator.translate(text, dest=dest_lang)
        return detected, translated

    def handle_translation(self, prompt: str) -> dict:
        try:
            # Parse translation request
            remaining = prompt[len('translate'):].strip()
            if ' to ' in remaining:
                text_part, lang_part = remaining.rsplit(' to ', 1)
                text_to_translate = text_part.strip().strip('"')
                target_lang = lang_part.strip()
            else:
                text_to_translate = remaining.strip('"')
                target_lang = st.session_state.get('target_lang', 'en')

            # Map target language to code
            lang_code = None
            for code, name in LANGUAGES.items():
                if target_lang.lower() in [name.lower(), code.lower()]:
                    lang_code = code
                    break
            lang_code = lang_code or 'en'

            # Run translation in sync context
            detected, translated = self.run_async(
                self.async_translate(text_to_translate, lang_code)
            )
            
            # Always generate TTS for translation requests if language is supported
            audio_path = None
            if lang_code in GTTS_LANG_CODES:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts = gTTS(translated.text, lang=GTTS_LANG_CODES[lang_code])
                    tts.save(tmp_file.name)
                    audio_path = tmp_file.name
                    st.session_state.temp_files.append(audio_path)

            return {
                'text': f"Detected Language: {LANGUAGES.get(detected.lang, 'Unknown')}\n\nTranslation: {translated.text}",
                'audio_path': audio_path
            }
        except Exception as e:
            logger.error(f"Translation error details: {str(e)}", exc_info=True)
            return {'text': f"Translation error: {str(e)}", 'audio_path': None}
        

class LanguagePipeline:
    def __init__(self):
        self.translator = Translator()
        self.detected_lang = None
        
    async def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            detection = await self.translator.detect(text)
            return detection.lang
        except Exception as e:
            st.error(f"Language detection error: {str(e)}")
            return 'en'  # Default to English on error
            
    async def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English if needed"""
        if source_lang == 'en':
            return text
        try:
            translation = await self.translator.translate(text, src=source_lang, dest='en')
            return translation.text
        except Exception as e:
            st.error(f"Translation to English error: {str(e)}")
            return text
            
    async def translate_response(self, text: str, target_lang: str) -> str:
        """Translate response back to original language"""
        if target_lang == 'en':
            return text
        try:
            translation = await self.translator.translate(text, src='en', dest=target_lang)
            return translation.text
        except Exception as e:
            st.error(f"Translation of response error: {str(e)}")
            return text

class EnhancedGeminiContextCache(GeminiContextCache):
    def __init__(self):
        super().__init__()
        self.language_pipeline = LanguagePipeline()
        
    async def process_multilingual_query(self, prompt: str, chat_history: list, use_files: bool = False) -> Dict:
        """Process queries in any language through the pipeline"""
        try:
            # Detect input language
            source_lang = await self.language_pipeline.detect_language(prompt)
            
            # Translate to English if needed
            english_prompt = await self.language_pipeline.translate_to_english(prompt, source_lang)
            
            # Get model response in English
            english_response = self.query_model(english_prompt, chat_history, use_files)
            
            # Translate response back to original language
            if source_lang != 'en':
                translated_text = await self.language_pipeline.translate_response(
                    english_response['text'], 
                    source_lang
                )
                english_response['text'] = translated_text
                
            # Add language info to response
            english_response['detected_language'] = LANGUAGES.get(source_lang, 'Unknown')
            
            return english_response
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            st.error(error_msg)
            return {'text': error_msg, 'audio_path': None, 'detected_language': 'Unknown'}

def recognize_speech(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None

def safe_delete(file_path):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Could not delete file {file_path}: {str(e)}")


async def create_compact_translation_dropdown(message_idx, original_text, detected_lang):
    translation_key = f"trans_{message_idx}"
    expand_key = f"expand_{message_idx}"
    
    # Initialize expand state if not present
    if expand_key not in st.session_state:
        st.session_state[expand_key] = False
    
    # Convert detected_lang to code if it's a full language name
    if detected_lang in LANGUAGES.values():
        detected_lang = [k for k, v in LANGUAGES.items() if v == detected_lang][0]
    
    col1, col2 = st.columns([5, 1])
    
    with col2:
        # Add a button to toggle translation options
        if st.button("Translate", key=f"btn_{message_idx}"):
            st.session_state[expand_key] = not st.session_state[expand_key]
        
        # Only show language selection if expanded
        if st.session_state[expand_key]:
            lang_options = {
                detected_lang: f" {LANGUAGES.get(detected_lang, 'Original')}",
                'hi': ' Hindi',
                'en': ' English'
            }
            
            # Remove duplicates if detected language is already English or Hindi
            if detected_lang in ['en', 'hi']:
                lang_options.pop(detected_lang)
            
            selected_lang = st.selectbox(
                "Select Language",
                options=list(lang_options.keys()),
                format_func=lambda x: lang_options[x],
                key=translation_key,
                label_visibility="collapsed"
            )
            
            if selected_lang != detected_lang:
                # Check if translation already exists in session state
                cached_translation = st.session_state.translations.get(f"{translation_key}_{selected_lang}")
                if cached_translation:
                    return cached_translation
                    
                try:
                    translator = Translator()
                    result = await translator.translate(
                        original_text,
                        src=detected_lang,
                        dest=selected_lang
                    )
                    # Cache the translation
                    st.session_state.translations[f"{translation_key}_{selected_lang}"] = result.text
                    return result.text
                except Exception as e:
                    logger.error(f"Translation error: {str(e)}")
                    return original_text
                
    return original_text

def main():
    st.set_page_config(
        page_title="AI Assistant with Speech & Translation",
        layout="wide"
    )
    
    # Initialize session state
    if 'cache' not in st.session_state:
        st.session_state.cache = EnhancedGeminiContextCache()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    if 'target_lang' not in st.session_state:
        st.session_state.target_lang = 'en'
    if 'current_cache' not in st.session_state:
        st.session_state.current_cache = None
    if 'system_instruction' not in st.session_state:
        st.session_state.system_instruction = "Analyze the provided files and answer questions about them."
    if 'ttl_minutes' not in st.session_state:
        st.session_state.ttl_minutes = 10
    if 'translations' not in st.session_state:
        st.session_state.translations = {}
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

    # Sidebar configuration (keep previous file and translation settings)
    with st.sidebar:
        # st.header("Configuration")
        # if st.session_state.cache.general_model:
        #     st.success("API Configured from environment variables!")
        # else:
        #     st.error("API key not found. Please set GOOGLE_API_KEY in your .env file")

        st.header("File Management")
        uploaded_files = st.file_uploader(
            "Upload files for context",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'doc', 'jpg', 'png', 'mp4']
        )
        
        if uploaded_files:
            cache_name = st.text_input("Cache Name", 
                                     value=f"cache_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            st.session_state.system_instruction = st.text_area(
                "System Instruction",
                value=st.session_state.system_instruction
            )
            st.session_state.ttl_minutes = st.number_input(
                "Cache Duration (minutes)", 
                min_value=1, 
                max_value=60, 
                value=10
            )
            
            if st.button("Process Files"):
                st.session_state.current_cache = st.session_state.cache.update_cache(
                    uploaded_files,
                    cache_name,
                    st.session_state.system_instruction,
                    st.session_state.ttl_minutes
                )
                if st.session_state.current_cache:
                    st.success(f"Using cache: {st.session_state.current_cache}")

        # st.header("Translation Settings")
        # st.session_state.target_lang = st.selectbox(
        #     "Target Language",
        #     options=list(LANGUAGES.keys()),
        #     format_func=lambda x: f"{x} - {LANGUAGES[x]}"
        # )
        # st.session_state.enable_tts = st.checkbox("Enable Text-to-Speech")

    # Main interface
    st.title("AI Assistant with Speech & Translation")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    st.markdown(msg["content"])
                    
                    # Only add translation dropdown for assistant messages
                    translated_text = st.session_state.cache.run_async(
                        create_compact_translation_dropdown(
                            idx,
                            msg["content"],
                            msg.get("detected_language", "en")[:2].lower()
                        )
                    )
                    
                    if translated_text != msg["content"]:
                        st.markdown(translated_text)
                    
                    if msg.get("audio"):
                        st.audio(msg["audio"], format="audio/mp3")
                else:
                    st.markdown(msg["content"])


    # Input container with voice recording
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            prompt = st.chat_input("Type or speak your message...")

        with col2:
            if not st.session_state.cache.general_model:
                st.button("🎤 Record", disabled=True, help="Please configure API key first")
            else:
                button_label = "🎤 Stop Recording" if st.session_state.is_recording else "🎤 Start Recording"
                if st.button(button_label):
                    try:
                        if st.session_state.is_recording:
                            # Stop recording
                            st.session_state.is_recording = False
                            audio_path = st.session_state.audio_recorder.stop_recording()
                            if audio_path:
                                st.session_state.temp_files.append(audio_path)
                                text = recognize_speech(audio_path)
                                if text:
                                    st.session_state.chat_history.append({"role": "user", "content": text})
                                    st.rerun()
                                else:
                                    st.error("Could not recognize speech. Please try again.")
                        else:
                            # Start recording
                            st.session_state.is_recording = True
                            st.session_state.audio_recorder.start_recording()
                            st.info("Recording... Press button again to stop")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error recording audio: {str(e)}")
                        st.session_state.is_recording = False

    # Handle text input
    if prompt:
        if not st.session_state.cache.general_model:
            st.error("Please configure your API key in the sidebar first")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Response generation (keep previous response handling)
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        user_input = st.session_state.chat_history[-1]["content"]
        
        with st.spinner("Processing..."):
            use_files = any(keyword in user_input.lower() for keyword in ['file', 'document', 'page'])
            
            # Use the new multilingual pipeline
            response = st.session_state.cache.run_async(
                st.session_state.cache.process_multilingual_query(
                    user_input,
                    st.session_state.chat_history,
                    use_files=use_files or bool(st.session_state.current_cache)
                )
            )
            
            if response['text']:
                original_text = response['text']
                detected_lang = response.get('detected_language', 'en').lower()[:2]
                
                response_data = {
                    "role": "assistant",
                    "content": original_text,
                    "detected_language": response.get('detected_language', 'Unknown')
                }
                if response.get('audio_path'):
                    response_data["audio"] = response['audio_path']
                
                st.session_state.chat_history.append(response_data)
                
                st.rerun()
    # Cleanup
    for file in st.session_state.temp_files:
        safe_delete(file)
    st.session_state.temp_files = []

if __name__ == "__main__":
    main()

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)