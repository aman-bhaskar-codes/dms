from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Camera
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    flip_horizontal: bool = True

    # Detection
    yolo_model: str = "yolov8x.pt"          # upgraded from yolov8n
    yolo_traffic_model: str = "models/traffic_dms_v5.onnx"  # fine-tuned
    mediapipe_model_complexity: int = 1     # 0=lite 1=full 2=heavy
    use_face_mesh_attention: bool = True

    # RAG & LLM
    ollama_base_url: str = "http://localhost:11434"
    primary_llm: str = "llama3.2:3b"
    fallback_llm: str = "phi4-mini"
    embed_model: str = "nomic-embed-text"
    rag_top_k: int = 5
    rag_rerank: bool = True

    # Voice
    whisper_model: str = "base.en"          # or "small.en" for accuracy
    whisper_device: str = "cpu"             # "cuda" if available
    tts_engine: str = "coqui"              # "pyttsx3" as fallback
    vad_threshold: float = 0.5

    # Memory
    db_path: str = "data/dms_v5.db"
    chroma_path: str = "data/chroma_v5"
    graph_db_path: str = "data/graph.db"    # NEW: SQLite graph
    memory_max_working_items: int = 200
    semantic_embed_interval_s: int = 300    # embed every 5 min

    # Alerts
    fatigue_warn_threshold: float = 55.0
    fatigue_crit_threshold: float = 72.0
    microsleep_ear_threshold: float = 0.18
    alert_cooldown_s: float = 8.0

    # UI
    web_port: int = 8080
    debug_mode: bool = False
    quit_key: int = ord('q')

    # Coaching
    coaching_interval_min: float = 5.0
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
