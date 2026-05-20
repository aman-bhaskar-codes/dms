# 🚗 Agentic Driver Monitoring System (DMS V3)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=white&color=3776AB)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?logo=opencv&logoColor=white&color=5C3EE8)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg?logo=google&logoColor=white&color=00C4CC)](https://google.github.io/mediapipe/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-red.svg?logo=ollama&logoColor=white&color=FF4500)](https://ollama.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg?logo=database&logoColor=white&color=A855F7)](https://www.trychroma.com/)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-blue.svg?logo=qt&logoColor=white&color=41CD52)](https://www.riverbankcomputing.com/software/pyqt/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-teal.svg?logo=fastapi&logoColor=white&color=009688)](https://fastapi.tiangolo.com/)

An elite, production-grade **Agentic Digital Twin Driver Monitoring System (DMS V3)** designed to run 100% offline. DMS V3 moves beyond reactive alerts to actively analyze driver fatigue, maintain multi-tier episodic/semantic memory, route driver intent with an offline RAG Voice Assistant, and orchestrate protective interventions using a multi-threaded multi-agent loop.

---

## 🌟 Key Features

*   **⚡ Sub-33ms Perception Pipeline:** Real-time facial landmark tracking using **MediaPipe Face Mesh (478 pts)** with per-landmark **Kalman Filter smoothing** for rock-solid jitter removal under extreme lighting variations.
*   **📊 9-Signal Composite Fatigue Index:** An advanced sensor-fusion engine computing a real-time `FatigueScore` (0-100) based on:
    *   Adaptive EAR (Eye Aspect Ratio) + Blink Velocity + Slow-blink ratio.
    *   Weighted PERCLOS (Percentage of Eye Closure) with predictive linear trend forecasting.
    *   3D Head Pose Yaw/Pitch/Roll + Head Sway oscillation + Nod frequency.
    *   Gaze Quality (Iris tracking, attention heatmap, saccade velocity).
    *   Non-contact rPPG (Remote Photoplethysmography) heart rate extraction from facial green-channel FFT.
    *   Landmark velocity bursts (micro-expression classifiers).
*   **🧠 Multi-Tier Unified Memory Stack:**
    *   **Working Memory (RAM):** Instantaneous session state.
    *   **Episodic Memory (SQLite):** SQL-indexed records of all mid-drive events, alerts, and continuous parameters.
    *   **Semantic Memory (ChromaDB Vector Store):** LLM-extracted behavioral patterns and session summaries embedded locally using `nomic-embed-text`.
    *   **Adaptive Driver Profiles:** Self-calibrating biometric thresholds that continuous adjust for drift, facial geometries, or lighting.
*   **🎙️ Offline RAG Voice Assistant:** A bidirectional conversational loop running **Faster-Whisper STT** for speech recognition, local **Ollama** (`llama3.2:3b`/`phi4-mini`) for RAG & Intent Routing, and local **Coqui TTS** for high-quality spoken voice generation.
*   **🤖 Multi-Agent Orchestration:** A multi-agent framework utilizing specialized agents (**Safety, Memory, Coaching, Prediction, Report**) to assess risks, coordinate warnings, analyze trends, and deliver pre-emptive suggestions.
*   **🖥️ Professional PyQt6 Dashboard & Web Console:** A dual-interface presentation layer featuring a beautiful PyQt6 analytics window with scrolled real-time charting, and a concurrent **FastAPI REST/WebSocket server** for remote fleet telemetry.

---

## 🏗️ System Architecture

```
  WEBCAM (30fps)                    MICROPHONE (16kHz)
       │                                  │
       ▼                                  ▼
┌─────────────┐                  ┌────────────────────┐
│ OpenCV      │                  │  WHISPER STT       │
│ Frame Queue │                  │  (faster-whisper)  │
└──────┬──────┘                  └────────┬───────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PERCEPTION LAYER  (Main Thread, <33ms)            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │     MediaPipe Face Mesh (478 pts) + Per-Landmark Kalman         │ │
│  └─────┬──────┬──────┬──────┬──────────────────────────────┬──────┘ │
│        │      │      │      │                              │        │
│        ▼      ▼      ▼      ▼                              ▼        │
│  ┌──────┐ ┌────┐ ┌────┐ ┌──────────────┐       ┌──────────────────┐ │
│  │ EAR  │ │MAR │ │HEAD│ │  GAZE V3     │       │  PERCLOS V3      │ │
│  │Blink  │ │Yawn│ │POSE│ │ Fixation     │       │  Weighted+       │ │
│  │Dyn.   │ │Freq│ │Sway│ │ Heatmap      │       │  Predictive      │ │
│  └──┬───┘ └──┬─┘ └──┬─┘ └──────┬───────┘       └──────────┬───────┘ │
│     └────────┴───────┴───────────┴──────────────────────────┘        │
│                                  │                                   │
│                                  ▼                                   │
│          ┌────────────────────────────────────────────────────┐      │
│          │           FATIGUE SCORE ENGINE V3                  │      │
│          │  9 signals → Composite Score 0-100 + Trend         │      │
│          └──────────────────────┬─────────────────────────────┘      │
│                                 │                                    │
│   ┌─────────────────────────────┼──────────────────────────────────┐ │
│   ▼                             ▼                         ▼        │ │
│ ┌────────────┐         ┌──────────────────┐      ┌──────────────┐  │ │
│ │ rPPG       │         │  YOLOv8 Thread   │      │  SCENE       │  │ │
│ │ HR+HRV     │         │ Phone·cup·belt   │      │ Light/shadow │  │ │
│ └─────┬──────┘         └─────────┬────────┘      └──────┬───────┘  │ │
│       └─────────────────────────┴──────────────────────┘           │ │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │     SENSOR FUSION BUS       │
                    └─────────────┬──────────────┘
                                  │
          ┌───────────────────────┼──────────────────────┐
          │                       │                      │
          ▼                       ▼                      ▼
┌──────────────────┐   ┌─────────────────────┐  ┌──────────────────────┐
│  ALERT ENGINE    │   │   AGENTIC LAYER     │  │  VOICE AGENT         │
│                  │   │                     │  │  (RAG + Ollama)      │
│  Escalation      │   │  ┌───────────────┐  │  │                      │
│  Ladder          │   │  │ ORCHESTRATOR  │  │  │  Whisper STT         │
│  TTS Priority    │   │  └──────┬────────┘  │  │  Intent Router       │
│  Flash pattern   │   │         ▼           │  │  ChromaDB Retrieval  │
│  Sound profiles  │   │ ┌────┐ ┌───┐ ┌────┐│  │  Ollama Generation   │
│  Haptic (USB)    │   │ │SAF │ │MEM│ │REP ││  │  Coqui TTS           │
│└────────┬─────────┘   │ │ETY │ │ORY│ │ORT ││  │  Session Context     │
          │             │ └─┬──┘ └─┬─┘ └──┬─┘│  │  └──────────┬───────────┘
          │             └───┼──────┼──────┼──┘  │             │
          │                 │      │      │     │             │
          └─────────────────┴──────┼──────┴─────┼─────────────┘
                                   │            │
           ┌───────────────────────▼────────────▼─────────────┐
           │                     MEMORY SYSTEM                │
           │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
           │  │ WORKING    │  │ EPISODIC   │  │  SEMANTIC  │  │
           │  │ (RAM dict) │  │ (SQLite)   │  │ (ChromaDB) │  │
           │  └────────────┘  └────────────┘  └────────────┘  │
           └───────────────────────┬──────────────────────────┘
                                   │
           ┌───────────────────────▼──────────────────────────┐
           │                  PRESENTATION LAYER              │
           │  ┌─────────────────────────────────────────────┐ │
           │  │         PyQt6 PROFESSIONAL DASHBOARD        │ │
           │  │   HUD Panel  ·  Analytics Charts  · Memory   │ │
           │  └─────────────────────────────────────────────┘ │
           │  ┌─────────────────────────────────────────────┐ │
           │  │             FastAPI WEB INTERFACE           │ │
           │  │   REST API  ·  WebSockets  · Remote Console │ │
           │  └─────────────────────────────────────────────┘ │
           └──────────────────────────────────────────────────┘
```

---

## 📈 Signal Pipeline & Weights

| Signal | Weight | Feature Description | Core Algorithm / Model |
| :--- | :---: | :--- | :--- |
| **EAR** | 22% | Eye Aspect Ratio tracking + Blinks | Soukupová & Čech + 1D Kalman |
| **PERCLOS** | 18% | Percentage of Eye Closure | Weighted rolling NHTSA ratio |
| **Blink Dynamics** | 13% | Closure velocity + slow blink ratios | Differential landmark tracking |
| **Head Sway** | 9% | Nod frequency + sway oscillation | Head geometry pose vector variance |
| **MAR** | 9% | 3D Mouth Aspect Ratio yawning | Landmark Euclidean ratios |
| **Gaze Quality** | 10% | Attention heatmap + saccade speed | Iris tracking + saccade classification |
| **rPPG** | 10% | heart rate & HRV-based stress | Forehead green-channel FFT |
| **Scene Context** | 5% | Glare & ambient illumination scaling | Thresholded luminosity analysis |
| **Micro-expression** | 4% | Dynamic emotional/facial twitches | High-frequency landmark velocity |

---

## 🛠️ Installation & Setup

### 📋 Prerequisites

*   **Operating System:** macOS (fully optimized), Linux, or Windows.
*   **Hardware:** Webcam (built-in or USB) + Microphone (for voice interaction).
*   **Ollama:** Installed and running locally.
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ```

### ⚡ Automatic Setup

Run the comprehensive automatic installation script which will set up directories, initialize environment configs, create the Python virtual environment, and install dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

### 🧠 Model Setup

Pull the necessary local models using Ollama:

```bash
# LLMs for Agent Orchestration, Intent Routing, and RAG
ollama pull llama3.2:3b
ollama pull phi4-mini

# Local Embeddings for semantic memory
ollama pull nomic-embed-text
```

---

## 🚀 Running the System

Activate the virtual environment and run the orchestrator:

```bash
# Activate virtual environment
source venv/bin/activate

# Launch in PyQt6 Dashboard mode (Default)
python main.py

# Launch in Headless Fleet Mode with REST API + WebSockets
python main.py --mode headless
```

Once running in dashboard mode:
1. The **HUD Panel** displays the real-time webcam feed overlay with iris tracking, head pose axes, and facial mesh outlines.
2. The **Analytics Panel** visualizes scrolling real-time graphs of your biometrics, rPPG signal, stress, and fatigue.
3. The **Memory Panel** allows you to review your driver profiles, search previous driving session logs, and chat directly with your offline Agentic Coach.

---

## 💬 Voice Agent Interaction Examples

The system continually monitors voice input via Safer-Whisper and processes commands locally:

*   **Biometric Inquiry:**  
    > *Driver:* "What is my fatigue score right now?"  
    > *Voice Assistant:* "Your fatigue score is currently 32, which is well within the safe range. Your heart rate is stable at 72 BPM."
*   **Historical Comparison:**  
    > *Driver:* "How am I doing compared to yesterday's drive?"  
    > *Voice Assistant:* "You are performing much better. Yesterday at this time, you showed a peak fatigue of 68 with three yawning events. Today, you are stable with zero alerts logged."
*   **Cognitive Support / Nudges:**  
    > *Driver:* "I'm starting to feel quite tired."  
    > *Voice Assistant:* "Noted. Your blink velocity has slowed down and your fatigue score is rising. There is a verified rest stop coming up in 3 miles. I recommend we pull over for a quick stretch."

---

## 📁 Repository Structure

*   [`perception/`](file:///Users/amanbhaskar/dms/perception) — Landmark trackers, EAR/MAR detectors, head pose estimation, gaze heatmaps, rPPG stress analyzer, and composite Fatigue score modules.
*   [`agents/`](file:///Users/amanbhaskar/dms/agents) — Multi-agent system orchestrator, Safety agent, Memory agent, Coaching agent, and Predictive agent modules, along with execution tools.
*   [`memory/`](file:///Users/amanbhaskar/dms/memory) — Core databases (SQLite & ChromaDB), memory managers, working memory dictionaries, and driver profiles.
*   [`voice_agent/`](file:///Users/amanbhaskar/dms/voice_agent) — Faster-Whisper, local intent router, local RAG pipeline, and Coqui/pyttsx3 speech generators.
*   [`dashboard/`](file:///Users/amanbhaskar/dms/dashboard) — Premium PyQt6 dashboard layouts, real-time scrolling charts, overlay widgets, and FastAPI WebSocket servers.
*   [`calibration/`](file:///Users/amanbhaskar/dms/calibration) — Adaptive calibration loop to baseline driver thresholds continuous over time.
*   [`main.py`](file:///Users/amanbhaskar/dms/main.py) — Core runner and mode selector.
*   [`config.py`](file:///Users/amanbhaskar/dms/config.py) — Universal configuration definitions.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
