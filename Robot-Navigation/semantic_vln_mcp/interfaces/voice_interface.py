"""
Voice Command Interface
========================
Part of SemanticVLN-MCP Framework

INNOVATIVE ADDITION: Voice control for robot navigation

Features:
- Speech-to-text using SpeechRecognition
- Wake word detection ("Hey Robot")
- Command parsing and forwarding
- Text-to-speech feedback

Author: SemanticVLN-MCP Team
"""

import time
from typing import Optional, Callable
import threading
import queue

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("Warning: SpeechRecognition not installed. Run: pip install SpeechRecognition")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not installed. Run: pip install pyttsx3")


class VoiceInterface:
    """
    Voice interface for robot control.
    
    Features:
    - Continuous listening for wake word
    - Speech recognition for commands
    - Text-to-speech feedback
    - Callback-based command handling
    """
    
    def __init__(self,
                 wake_word: str = "hey robot",
                 language: str = "en-US",
                 callback: Optional[Callable[[str], None]] = None):
        """
        Initialize voice interface.
        
        Args:
            wake_word: Word/phrase to activate listening
            language: Speech recognition language
            callback: Function to call with recognized commands
        """
        self.wake_word = wake_word.lower()
        self.language = language
        self.callback = callback
        
        # Speech recognition
        self.recognizer = None
        self.microphone = None
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            try:
                self.microphone = sr.Microphone()
                print("Microphone initialized")
            except Exception as e:
                print(f"Warning: Could not initialize microphone: {e}")
                self.microphone = None
        
        # Text-to-speech
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                print("TTS engine initialized")
            except Exception as e:
                print(f"Warning: Could not initialize TTS: {e}")
        
        # State
        self.listening = False
        self.awaiting_command = False
        self.command_queue = queue.Queue()
        self.listen_thread = None
    
    def speak(self, text: str):
        """Speak text using TTS."""
        print(f"[VOICE] Speaking: {text}")
        
        if self.tts_engine is not None:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def listen_once(self, timeout: float = 5.0) -> Optional[str]:
        """
        Listen for a single speech input.
        
        Args:
            timeout: Maximum time to wait for speech
            
        Returns:
            Recognized text or None
        """
        if self.recognizer is None or self.microphone is None:
            return None
        
        try:
            with self.microphone as source:
                print("[VOICE] Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"[VOICE] Recognized: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("[VOICE] Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"[VOICE] Recognition service error: {e}")
            return None
        except Exception as e:
            print(f"[VOICE] Error: {e}")
            return None
    
    def _listen_loop(self):
        """Background listening loop."""
        while self.listening:
            text = self.listen_once(timeout=2.0)
            
            if text is None:
                continue
            
            if not self.awaiting_command:
                # Check for wake word
                if self.wake_word in text:
                    self.speak("Yes, I'm listening")
                    self.awaiting_command = True
            else:
                # Process command
                # Remove wake word if repeated
                command = text.replace(self.wake_word, "").strip()
                
                if command:
                    self.speak(f"Processing: {command}")
                    self.command_queue.put(command)
                    
                    if self.callback:
                        self.callback(command)
                
                self.awaiting_command = False
    
    def start_listening(self):
        """Start background listening."""
        if self.recognizer is None or self.microphone is None:
            print("Cannot start listening: no microphone")
            return
        
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        self.speak(f"Voice control activated. Say '{self.wake_word}' to give a command.")
        print(f"[VOICE] Started listening for wake word: '{self.wake_word}'")
    
    def stop_listening(self):
        """Stop background listening."""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=3.0)
        print("[VOICE] Stopped listening")
    
    def get_command(self, block: bool = False, timeout: float = None) -> Optional[str]:
        """
        Get a command from the queue.
        
        Args:
            block: Wait for command if queue is empty
            timeout: Maximum wait time if blocking
            
        Returns:
            Command string or None
        """
        try:
            return self.command_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def process_text_command(self, text: str):
        """Process a text command (for testing without mic)."""
        if self.callback:
            self.callback(text)
        self.command_queue.put(text)


# Standalone test
if __name__ == "__main__":
    print("Testing Voice Interface...")
    
    def handle_command(cmd):
        print(f"Command received: {cmd}")
    
    voice = VoiceInterface(callback=handle_command)
    
    # Test TTS
    voice.speak("Hello! Voice interface is ready.")
    
    # Test listening (if microphone available)
    if voice.microphone:
        print("\nSay something (5 seconds)...")
        text = voice.listen_once(timeout=5)
        if text:
            print(f"You said: {text}")
    else:
        print("\nNo microphone available. Testing text input...")
        voice.process_text_command("Navigate to the kitchen")
        
        cmd = voice.get_command()
        print(f"Queued command: {cmd}")
    
    print("\nVoice Interface test complete!")
