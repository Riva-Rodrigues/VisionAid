from assistant import VisualAssistant

def main():
    try:
        assistant = VisualAssistant()
        assistant.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("\nMake sure you have:")
        print("1. A working webcam")
        print("2. Internet connection (for initial model download)")
        print("3. Microphone access")
        print("4. All required packages installed")

if __name__ == "__main__":
    
    main()