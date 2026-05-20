from voice_agent.stt_engine import STTEngine
from voice_agent.tts_engine import TTSEngine
from voice_agent.rag_pipeline import RAGPipeline
from memory.semantic_memory import SemanticMemory
import threading
import time

class VoiceAgent:
    """
    Main controller for the RAG-powered Voice Agent.
    Handles listening (STT), reasoning (RAG + Ollama), and speaking (TTS).
    """
    def __init__(self, semantic_memory: SemanticMemory):
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.rag = RAGPipeline(semantic_memory)
        self.is_listening = False
        
    def listen_and_respond(self, driver_id: str, audio_file_path: str, current_metrics: dict):
        """
        Processes an audio file containing the driver's speech and responds.
        Can be run in a separate thread to not block the main loop.
        """
        if self.is_listening:
            return
            
        self.is_listening = True
        try:
            # 1. Transcribe
            print("[VoiceAgent] Processing audio...")
            text = self.stt.transcribe(audio_file_path)
            
            if not text:
                print("[VoiceAgent] No speech detected.")
                return
                
            # 2. RAG Generation
            print(f"[VoiceAgent] Thinking about: '{text}'")
            response = self.rag.generate_response(driver_id, text, current_metrics)
            
            # 3. Speak
            self.tts.speak(response)
            
        except Exception as e:
            print(f"[VoiceAgent] Error: {e}")
        finally:
            self.is_listening = False
