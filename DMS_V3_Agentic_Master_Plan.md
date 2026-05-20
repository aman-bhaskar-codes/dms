# 🚗 Driver Monitoring System V3 — Agentic AI Architecture
## Complete Professional Implementation Plan

> **Production-Ready · 100% Offline · Agentic · RAG Voice Agent · Long-Term Memory**  
> **V3 Stack: MediaPipe · OpenCV · YOLOv8 · rPPG · Kalman · Ollama · ChromaDB · Whisper · FastAPI · PyQt6 · LangGraph**

---

## Executive Summary

DMS V3 is a **production-grade, agentic driver monitoring system** that goes far beyond reactive alerting. It reasons about driver state across time, maintains long-term memory per driver, orchestrates specialized AI agents, and provides a fully voice-interactive interface backed by a local RAG (Retrieval-Augmented Generation) pipeline.

| Layer | V1 | V2 | V3 |
|---|---|---|---|
| Detection | EAR, MAR, head, gaze, PERCLOS | +rPPG, blink dynamics, fatigue score, Kalman | +Micro-expression, EMG-sim, scene context |
| AI | Ollama 5-min report | +Contextual mid-drive tips | **Agentic reasoning, planning, self-reflection** |
| Memory | None | Driver profile JSON | **Vector store + episodic + semantic + working** |
| Voice | pyttsx3 TTS alerts | pyttsx3 escalating alerts | **Full bidirectional RAG voice agent (Whisper + Coqui)** |
| UI | OpenCV HUD | +Matplotlib analytics window | **PyQt6 professional dashboard + web interface** |
| Architecture | Sequential | Threaded | **Multi-agent with LangGraph orchestration** |

---

## 1. V1 vs V2 Comparative Analysis

### V1 Strengths & Weaknesses
```
✅ STRENGTHS
  - Clean modular structure (detectors/, alerts/, dashboard/)
  - Solid EAR/MAR/PERCLOS/gaze/head-pose pipeline
  - SQLite logging with Ollama 5-min reports
  - Simple to understand and modify

❌ WEAKNESSES
  - Fixed thresholds — fails spectacularly with glasses, lighting, face geometry
  - No memory — every session starts blind; no learning from history
  - Reactive-only — waits for event, then alerts; no predictive capability
  - Single-threaded YOLO — blocks main loop every 5th frame
  - TTS = pyttsx3 alert barks; no conversational intelligence
  - Ollama integration = fire-and-forget prompt; no context, no conversation
  - No UI beyond OpenCV overlay + matplotlib popup
```

### V2 Improvements & Remaining Gaps
```
✅ ADDED IN V2
  - Kalman filter per-landmark — massive jitter reduction
  - Adaptive 30s calibration → per-driver thresholds
  - rPPG heart rate from camera (green channel FFT)
  - 7-signal composite FatigueScore 0–100
  - Blink dynamics (closure velocity + slow-blink ratio)
  - Head sway/jerk detection + microsleep wake-up detection
  - Fixation map + attention heatmap
  - Threaded YOLO — no longer blocking
  - TTS escalation ladder with priority queue
  - Structured JSON session export

❌ REMAINING GAPS IN V2
  - Memory still session-scoped (driver_profiles JSON is just thresholds)
  - AI reporter still stateless — doesn't know prior sessions
  - No voice input — driver cannot ask questions
  - No agentic planning — system cannot adjust strategy mid-session
  - Dashboard is still OpenCV — no proper GUI framework
  - No RAG — AI answers from model weights only, not driver's history
  - No scene understanding — ignores road context (day/night/weather)
  - Prediction = none — all reactive, never anticipatory
  - No fleet support — single-driver only
  - No REST API — cannot integrate with fleet management systems
```

---

## 2. V3 Design Principles

1. **Agentic First** — AI is not a report generator; it is an orchestrator that *plans*, *delegates*, *reflects*, and *adapts*
2. **Memory as Foundation** — every insight is stored, indexed, and retrieved; the system gets smarter with each session
3. **RAG over Hallucination** — voice agent answers are grounded in actual driver history, not model guesses  
4. **Offline Sovereign** — zero cloud dependency; all models run via local Ollama
5. **Professional UI** — PyQt6 primary dashboard + FastAPI web interface for fleet/remote monitoring
6. **Latency Budget Preserved** — agentic layers run in background threads; camera→detection pipeline stays <33ms
7. **Fail-Safe Degradation** — every AI component can fail gracefully; core safety alerts always work

---

## 3. V3 Full Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║              DRIVER MONITORING SYSTEM V3 — AGENTIC ARCHITECTURE                 ║
║         Production-Ready · 100% Offline · Multi-Agent · RAG Voice               ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  WEBCAM (30fps)                    MICROPHONE (16kHz)
       │                                  │
       ▼                                  ▼
┌─────────────┐                  ┌────────────────────┐
│ OpenCV      │                  │  WHISPER STT       │
│ Frame Queue │                  │  (whisper.cpp/     │
│ Thread-safe │                  │   faster-whisper)  │
└──────┬──────┘                  └────────┬───────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PERCEPTION LAYER  (Main Thread, <33ms)            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │     MediaPipe Face Mesh (478 pts) + Per-Landmark Kalman V2      │ │
│  └─────┬──────┬──────┬──────┬──────────────────────────────┬──────┘ │
│        │      │      │      │                              │        │
│        ▼      ▼      ▼      ▼                              ▼        │
│  ┌──────┐ ┌────┐ ┌────┐ ┌──────────────┐       ┌──────────────────┐ │
│  │ EAR  │ │MAR │ │HEAD│ │  GAZE V3     │       │  PERCLOS V3      │ │
│  │ V3   │ │ V3 │ │POSE│ │ Fixation     │       │  Weighted+       │ │
│  │Blink │ │ 3D │ │ V3 │ │ Heatmap      │       │  Predictive      │ │
│  │Dyn.  │ │Yawn│ │Sway│ │ Saccade      │       │  Trend           │ │
│  │Micro │ │Freq│ │Jerk│ │ Entropy      │       │                  │ │
│  └──┬───┘ └──┬─┘ └──┬─┘ └──────┬───────┘       └──────────┬───────┘ │
│     │        │       │           │                          │        │
│     └────────┴───────┴───────────┴──────────────────────────┘        │
│                                  │                                   │
│                                  ▼                                   │
│          ┌────────────────────────────────────────────────────┐      │
│          │           FATIGUE SCORE ENGINE V3                  │      │
│          │  9 signals → Composite Score 0-100 + Trend         │      │
│          │  Signals: EAR · PERCLOS · Blink · Sway · MAR ·    │      │
│          │           Gaze · rPPG · Micro-expr · Saccade       │      │
│          │  Prediction: LinearRegression next-5-min forecast  │      │
│          └──────────────────────┬─────────────────────────────┘      │
│                                 │                                    │
│   ┌─────────────────────────────┼──────────────────────────────────┐ │
│   │                             │                                  │ │
│   ▼                             ▼                         ▼        │ │
│ ┌────────────┐         ┌──────────────────┐      ┌──────────────┐  │ │
│ │ rPPG V3   │         │  YOLOv8 Thread   │      │  SCENE V3   │  │ │
│ │ HR+HRV    │         │ Phone·cup·belt   │      │ Light/shadow│  │ │
│ │ Stress    │         │ Object tracking  │      │ Day/night   │  │ │
│ │ Coherence │         │ Velocity         │      │ Detection   │  │ │
│ └─────┬──────┘         └─────────┬────────┘      └──────┬───────┘  │ │
│       └─────────────────────────┴──────────────────────┘           │ │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │     SENSOR FUSION BUS       │
                    │  Normalized metrics dict    │
                    │  Published at 30fps         │
                    └─────────────┬──────────────┘
                                  │
          ┌───────────────────────┼──────────────────────┐
          │                       │                      │
          ▼                       ▼                      ▼
┌──────────────────┐   ┌─────────────────────┐  ┌──────────────────────┐
│  ALERT ENGINE    │   │   AGENTIC LAYER      │  │  VOICE AGENT         │
│  V3              │   │   (LangGraph)        │  │  (RAG + Ollama)      │
│                  │   │                      │  │                      │
│  Escalation      │   │  ┌───────────────┐  │  │  Whisper STT         │
│  Ladder          │   │  │ ORCHESTRATOR  │  │  │  Intent Router       │
│  TTS Priority    │   │  │  Agent        │  │  │  ChromaDB Retrieval  │
│  Haptic (USB)    │   │  └──────┬────────┘  │  │  Ollama Generation   │
│  Sound profiles  │   │         │           │  │  Coqui TTS           │
│  Flash pattern   │   │  ┌──────┼────────┐  │  │  Session Context     │
└────────┬─────────┘   │  │      │        │  │  └──────────┬───────────┘
         │             │  ▼      ▼        ▼  │             │
         │             │ ┌────┐ ┌───┐ ┌────┐│             │
         │             │ │SAF │ │MEM│ │REP ││             │
         │             │ │ETY │ │ORY│ │ORT ││             │
         │             │ │AGT │ │AGT│ │AGT ││             │
         │             │ └─┬──┘ └─┬─┘ └──┬─┘│             │
         │             │   │      │      │  │             │
         │             └───┼──────┼──────┼──┘             │
         │                 │      │      │                 │
         └─────────────────┴──────┼──────┴─────────────────┘
                                  │
          ┌───────────────────────▼──────────────────────────┐
          │              MEMORY SYSTEM V3                     │
          │                                                   │
          │  ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
          │  │ WORKING    │ │ EPISODIC   │ │  SEMANTIC    │ │
          │  │ MEMORY     │ │ MEMORY     │ │  MEMORY      │ │
          │  │ (RAM dict) │ │ (SQLite)   │ │  (ChromaDB)  │ │
          │  │ Current    │ │ All events │ │  Embeddings  │ │
          │  │ session    │ │ Sessions   │ │  Nomic-embed │ │
          │  │ metrics    │ │ Patterns   │ │  RAG source  │ │
          │  └────────────┘ └────────────┘ └──────────────┘ │
          │                                                   │
          │  ┌────────────────────────────────────────────┐  │
          │  │          DRIVER PROFILE STORE              │  │
          │  │  Biometric calibration · Risk scores ·    │  │
          │  │  Behavioral patterns · Session history ·  │  │
          │  │  Recommendations history                   │  │
          │  └────────────────────────────────────────────┘  │
          └──────────────────────┬───────────────────────────┘
                                 │
          ┌──────────────────────▼────────────────────────────┐
          │               PRESENTATION LAYER                  │
          │                                                   │
          │  ┌──────────────────────────────────────────┐    │
          │  │        PyQt6 PROFESSIONAL DASHBOARD       │    │
          │  │                                           │    │
          │  │  ┌──────────┐  ┌──────────┐  ┌────────┐  │    │
          │  │  │ HUD      │  │ANALYTICS │  │ MEMORY │  │    │
          │  │  │ Panel    │  │ Panel    │  │ Panel  │  │    │
          │  │  │ Live feed│  │ Charts   │  │ Browser│  │    │
          │  │  │ Gauges   │  │ History  │  │ Agent  │  │    │
          │  │  │ Alerts   │  │ Trends   │  │ Chat   │  │    │
          │  │  └──────────┘  └──────────┘  └────────┘  │    │
          │  └──────────────────────────────────────────┘    │
          │                                                   │
          │  ┌──────────────────────────────────────────┐    │
          │  │         FastAPI WEB INTERFACE             │    │
          │  │  REST API · WebSocket live feed ·         │    │
          │  │  Fleet dashboard · Remote monitoring      │    │
          │  └──────────────────────────────────────────┘    │
          └───────────────────────────────────────────────────┘
```

---

## 4. V3 Directory Structure

```
dms_v3/
├── main.py                          # Entry point — CLI/GUI mode selector
├── config.py                        # Unified V3 configuration
├── requirements.txt
├── setup.sh                         # Full V3 install script
├── docker-compose.yml               # Optional containerized deployment
│
├── perception/                      # ── PERCEPTION LAYER ──
│   ├── __init__.py
│   ├── camera_manager.py            # Thread-safe frame queue + camera lifecycle
│   ├── face_detector.py             # MediaPipe + per-landmark Kalman V3
│   ├── ear_detector.py              # EAR + blink dynamics + micro-expression
│   ├── mar_detector.py              # MAR + 3D yawn + frequency analysis
│   ├── head_pose.py                 # solvePnP + sway + jerk + nodding frequency
│   ├── gaze_tracker.py              # Iris + fixation + heatmap + saccade velocity
│   ├── perclos.py                   # Weighted + predictive PERCLOS
│   ├── rppg.py                      # HR + HRV + stress coherence
│   ├── object_detector.py           # YOLOv8 threaded: phone/belt/cup/eyes-closed
│   ├── scene_analyzer.py            # NEW: lighting/day-night/glare/distraction zones
│   └── fatigue_score.py             # 9-signal composite + trend prediction
│
├── agents/                          # ── AGENTIC LAYER ──
│   ├── __init__.py
│   ├── orchestrator.py              # LangGraph main agent graph
│   ├── safety_agent.py              # Real-time risk assessment + intervention
│   ├── memory_agent.py              # Memory CRUD + retrieval + pattern detection
│   ├── report_agent.py              # Session/fleet report generation
│   ├── coaching_agent.py            # Personalized driving improvement coach
│   ├── prediction_agent.py          # Next-state forecasting + pre-emptive alerts
│   └── tools/
│       ├── __init__.py
│       ├── memory_tools.py          # save_memory, query_memory, get_driver_stats
│       ├── alert_tools.py           # trigger_alert, escalate, dismiss
│       ├── report_tools.py          # generate_report, export_pdf, email_report
│       └── sensor_tools.py          # get_current_metrics, get_trend, get_history
│
├── memory/                          # ── MEMORY SYSTEM ──
│   ├── __init__.py
│   ├── working_memory.py            # In-RAM session state (current metrics)
│   ├── episodic_memory.py           # SQLite: events, sessions, timelines
│   ├── semantic_memory.py           # ChromaDB: embeddings, RAG source
│   ├── driver_profile.py            # Biometric + behavioral driver profile
│   └── memory_manager.py           # Unified API for all memory subsystems
│
├── voice_agent/                     # ── VOICE AGENT ──
│   ├── __init__.py
│   ├── stt_engine.py                # Whisper STT (faster-whisper local)
│   ├── tts_engine.py                # Coqui TTS + pyttsx3 fallback
│   ├── intent_router.py             # Classify: query / command / conversation
│   ├── rag_pipeline.py              # ChromaDB retrieval + Ollama generation
│   ├── context_manager.py           # Maintain conversation history (5-turn)
│   └── voice_agent.py               # Main voice agent loop
│
├── alerts/                          # ── ALERT SYSTEM ──
│   ├── __init__.py
│   ├── alert_engine.py              # Escalation + priority queue + suppression
│   ├── alert_profiles.py            # Urban/highway/night/fleet profiles
│   └── notification_router.py       # TTS / sound / visual / webhook / push
│
├── dashboard/                       # ── UI LAYER ──
│   ├── __init__.py
│   ├── pyqt/
│   │   ├── main_window.py           # PyQt6 main window (3-panel layout)
│   │   ├── hud_panel.py             # Live feed + overlays + gauges
│   │   ├── analytics_panel.py       # Real-time charts (pyqtgraph)
│   │   ├── memory_panel.py          # Memory browser + agent chat
│   │   ├── session_panel.py         # Session history + replay
│   │   └── widgets/
│   │       ├── fatigue_gauge.py     # Circular gauge widget
│   │       ├── signal_chart.py      # Scrolling waveform widget
│   │       ├── heatmap_widget.py    # Attention heatmap overlay
│   │       ├── agent_chat.py        # Chat interface for agent interaction
│   │       └── driver_card.py       # Driver profile summary card
│   └── web/
│       ├── api_server.py            # FastAPI REST + WebSocket server
│       ├── static/                  # React frontend (built)
│       └── schemas.py               # Pydantic models
│
├── calibration/
│   ├── __init__.py
│   ├── driver_calibration.py        # 30s adaptive baseline + enhanced metrics
│   └── auto_tune.py                 # NEW: background continuous re-calibration
│
├── data/
│   ├── chroma_db/                   # ChromaDB vector store (persistent)
│   ├── sessions.db                  # SQLite episodic memory
│   ├── driver_profiles/             # Per-driver calibration + behavioral profiles
│   ├── models/                      # Local model caches
│   │   ├── whisper/                 # faster-whisper model
│   │   └── tts/                     # Coqui TTS model
│   └── reports/                     # Generated session + fleet reports
│
├── tests/
│   ├── test_perception.py
│   ├── test_memory.py
│   ├── test_voice_agent.py
│   ├── test_agents.py
│   └── fixtures/                    # Test video clips, expected outputs
│
└── docs/
    ├── API.md
    ├── DEPLOYMENT.md
    └── TUNING.md
```

---

## 5. Signal Pipeline V3

```
Signal              Weight   Algorithm                          New in V3
────────────────────────────────────────────────────────────────────────────────
EAR (Kalman)          22%    Soukupová & Čech + 1D Kalman      Micro-expression
PERCLOS (wtd)         18%    NHTSA + adaptive baseline          Predictive trend
Blink Dynamics        13%    Closure velocity + slow ratio      Neural classifier
Head Sway              9%    Jerk + oscillation + nod freq      Sway entropy
MAR / Yawn             9%    3D mouth ratio + frequency         Duration model
Gaze Quality          10%    Fixation + heatmap + saccade       Saccade velocity
rPPG HR+HRV           10%    Green FFT + HRV analysis           Stress coherence
Scene Context          5%    NEW: lighting + glare              Adaptive weight
Micro-expression       4%    NEW: landmark velocity burst        Surprise/fear
────────────────────────────────────────────────────────────────────────────────
FatigueScore V3     0–100    Weighted composite + ML regression
Prediction                   Linear regression 5-min forecast
```

---

## 6. Agentic System Design

### 6.1 Agent Graph (LangGraph)

```
                    ┌─────────────────────────────────┐
                    │         ORCHESTRATOR             │
                    │  Receives: sensor_fusion_bus     │
                    │  State: driver_state, history    │
                    │  Decides: which agents to invoke │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼──────────────────────────┐
          │                        │                          │
          ▼                        ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐   ┌──────────────────────┐
│  SAFETY AGENT   │    │    MEMORY AGENT       │   │   PREDICTION AGENT   │
│                 │    │                       │   │                      │
│  Tools:         │    │  Tools:               │   │  Tools:              │
│  · trigger_alert│    │  · save_event         │   │  · forecast_fatigue  │
│  · escalate     │    │  · query_history      │   │  · pattern_detect    │
│  · voice_warn   │    │  · update_profile     │   │  · risk_trajectory   │
│  · suggest_break│    │  · embed_insight      │   │  · alert_preemptive  │
│                 │    │  · get_similar_events │   │                      │
│  Triggers on:   │    │                       │   │  Runs every: 30s     │
│  fatigue > 45   │    │  Runs every: event    │   │  Horizon: 5 minutes  │
│  any critical   │    │  + every 60s bulk     │   │                      │
└─────────────────┘    └──────────────────────┘   └──────────────────────┘
          │                        │                          │
          └────────────────────────┼──────────────────────────┘
                                   │
          ┌────────────────────────┼──────────────────────────┐
          │                        │                          │
          ▼                        ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐   ┌──────────────────────┐
│  COACHING AGENT │    │    REPORT AGENT       │   │    VOICE AGENT       │
│                 │    │                       │   │                      │
│  Tools:         │    │  Tools:               │   │  Tools:              │
│  · analyze_trip │    │  · gen_session_report │   │  · listen_whisper    │
│  · get_patterns │    │  · gen_fleet_report   │   │  · route_intent      │
│  · suggest_tips │    │  · export_pdf         │   │  · rag_retrieve      │
│  · track_improve│    │  · compare_sessions   │   │  · ollama_generate   │
│                 │    │                       │   │  · speak_coqui       │
│  Runs: end-of-  │    │  Runs: end-of-session │   │  · maintain_context  │
│  session        │    │  + on-demand          │   │                      │
└─────────────────┘    └──────────────────────┘   └──────────────────────┘
```

### 6.2 Orchestrator State Machine

```python
# agents/orchestrator.py  — State & transition logic

AgentState = TypedDict('AgentState', {
    'driver_id': str,
    'session_id': str,
    'metrics': dict,          # Latest sensor fusion output
    'fatigue_score': float,
    'fatigue_level': str,     # normal / mild / warning / critical
    'active_alerts': list,
    'last_agent': str,
    'reasoning': str,         # Agent's chain-of-thought
    'memory_context': str,    # Retrieved relevant memories
    'action_taken': str,
    'turn': int,
})

# Graph edges (conditional routing)
def route(state: AgentState) -> str:
    fs = state['fatigue_score']
    if fs >= 70:   return 'safety_agent'    # critical — immediate intervention
    if fs >= 45:   return 'prediction_agent' # warning — forecast trajectory
    if fs >= 25:   return 'coaching_agent'  # mild — personalized nudge
    return 'memory_agent'                   # normal — background logging
```

### 6.3 Safety Agent — Full Reasoning Loop

```python
# agents/safety_agent.py

SAFETY_SYSTEM_PROMPT = """
You are an expert driver safety AI operating in real-time.
You have access to the driver's current biometric data and their full driving history.

Your job:
1. ASSESS the current risk level using provided metrics + memory context
2. DECIDE the most appropriate intervention (if any)
3. EXPLAIN your reasoning in 1 sentence
4. EXECUTE the intervention via the available tools

Rules:
- Be direct, brief, non-alarming unless critical
- Consider the driver's baseline from their profile
- Compare to similar past events in their history
- A drowsy driver who has pulled over before needs a harder push
- Never alert for conditions that are normal for this driver (from profile)

Available tools: trigger_alert, voice_warn, suggest_break, escalate, dismiss
"""

# Agent has access to:
# · Current metrics (all 9 signals)
# · Last 5 safety events this session
# · Driver's historical risk profile from ChromaDB
# · Predicted fatigue in next 5 minutes
```

---

## 7. Memory System V3 — Full Design

### 7.1 Memory Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEMORY HIERARCHY                             │
│                                                                     │
│  WORKING MEMORY (RAM)                                               │
│  ─────────────────────                                              │
│  Scope: Current session (lost on session end)                       │
│  Content: Live metrics, active alerts, agent state, chat context    │
│  Access: O(1) dict lookup                                           │
│  Size: ~100KB per session                                           │
│                                                                     │
│  EPISODIC MEMORY (SQLite)                                           │
│  ──────────────────────────                                         │
│  Scope: Permanent, all sessions ever                                │
│  Content: Events, sessions, frames, calibrations, recommendations   │
│  Access: SQL queries, indexed by driver_id + timestamp              │
│  Size: ~10MB/1000 hours of driving                                  │
│                                                                     │
│  SEMANTIC MEMORY (ChromaDB)                                         │
│  ──────────────────────────                                         │
│  Scope: Permanent, indexed for similarity search                    │
│  Content: Session summaries, insights, patterns, recommendations    │
│           Embedded with: nomic-embed-text (via Ollama)              │
│  Access: Vector similarity search (cosine)                          │
│  RAG Source: Voice agent queries this first                         │
│  Size: ~50MB/1000 sessions                                          │
│                                                                     │
│  DRIVER PROFILE (JSON + SQLite)                                     │
│  ──────────────────────────────                                     │
│  Scope: Per-driver, permanent                                       │
│  Content:                                                           │
│    · Biometric calibration (EAR baseline, head geometry)            │
│    · Behavioral fingerprint (typical blink rate, yaw range)         │
│    · Risk scores (rolling 30-day fatigue trend)                     │
│    · Session statistics (avg fatigue, worst sessions)               │
│    · Personalized thresholds (learned, not manual)                  │
│    · Coaching progress (which suggestions were acted on)            │
│    · Fleet metadata (vehicle ID, shift info)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 ChromaDB Collections

```python
# memory/semantic_memory.py

COLLECTIONS = {
    'session_insights': {
        'description': 'End-of-session AI summaries + key findings',
        'embedding_model': 'nomic-embed-text',
        'metadata_fields': ['driver_id', 'session_id', 'date', 'fatigue_peak',
                            'alert_count', 'risk_level'],
    },
    'safety_events': {
        'description': 'Individual safety-relevant events with context',
        'embedding_model': 'nomic-embed-text',
        'metadata_fields': ['driver_id', 'event_type', 'severity',
                            'fatigue_score', 'time_of_day', 'resolved_how'],
    },
    'driver_patterns': {
        'description': 'Detected behavioral patterns per driver',
        'embedding_model': 'nomic-embed-text',
        'metadata_fields': ['driver_id', 'pattern_type', 'frequency',
                            'conditions', 'first_seen', 'last_seen'],
    },
    'recommendations': {
        'description': 'All coaching recommendations + outcomes',
        'embedding_model': 'nomic-embed-text',
        'metadata_fields': ['driver_id', 'recommendation', 'given_at',
                            'followed', 'impact_score'],
    },
}

# Embedding pipeline
def embed_and_store(collection: str, text: str, metadata: dict, doc_id: str):
    """
    1. Call Ollama: POST /api/embeddings {model: 'nomic-embed-text', prompt: text}
    2. Get embedding vector (768-dim)
    3. Store in ChromaDB collection with metadata
    """
```

### 7.3 SQLite Schema V3

```sql
-- memory/episodic_memory.py  — Full schema

-- Driver registry
CREATE TABLE drivers (
    driver_id     TEXT PRIMARY KEY,
    name          TEXT,
    created_at    REAL,
    last_seen     REAL,
    total_sessions INTEGER DEFAULT 0,
    total_hours   REAL DEFAULT 0.0,
    risk_tier     TEXT DEFAULT 'unknown'   -- low/medium/high/critical
);

-- Session records
CREATE TABLE sessions (
    session_id    TEXT PRIMARY KEY,
    driver_id     TEXT REFERENCES drivers(driver_id),
    started_at    REAL NOT NULL,
    ended_at      REAL,
    duration_min  REAL,
    avg_fatigue   REAL,
    peak_fatigue  REAL,
    alert_count   INTEGER DEFAULT 0,
    phone_events  INTEGER DEFAULT 0,
    ai_report     TEXT,           -- Full Ollama session report
    risk_score    REAL,           -- Computed end-of-session
    notes         TEXT
);

-- Frame-level metrics (sampled every 1s, not 30fps — storage efficient)
CREATE TABLE frame_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT REFERENCES sessions(session_id),
    timestamp     REAL,
    ear           REAL, perclos REAL, mar REAL,
    yaw           REAL, pitch REAL, roll REAL,
    gaze_x        REAL, gaze_y REAL,
    hr_bpm        REAL, hrv_sdnn REAL,
    fatigue_score REAL,
    slow_blink_r  REAL, blink_rate REAL
);

-- Event log
CREATE TABLE events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT REFERENCES sessions(session_id),
    driver_id     TEXT,
    timestamp     REAL,
    event_type    TEXT,   -- drowsy_warn/drowsy_crit/yawn/head_nod/phone/break_taken/etc
    severity      TEXT,   -- info/warning/critical
    fatigue_score REAL,
    agent_action  TEXT,   -- What the safety agent decided
    agent_reason  TEXT,   -- Agent's reasoning (chain-of-thought)
    dismissed_by  TEXT    -- driver/auto/timeout
);

-- Coaching recommendations
CREATE TABLE recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT,
    driver_id     TEXT,
    given_at      REAL,
    recommendation TEXT,
    category      TEXT,  -- break/hydration/sleep/route/habit
    followed      INTEGER DEFAULT 0,
    impact_notes  TEXT
);

-- Memory agent observations
CREATE TABLE memory_observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id     TEXT,
    observed_at   REAL,
    pattern_type  TEXT,
    description   TEXT,
    confidence    REAL,
    chroma_id     TEXT   -- Cross-reference to ChromaDB document
);

-- Indexes for common queries
CREATE INDEX idx_events_driver   ON events(driver_id, timestamp);
CREATE INDEX idx_sessions_driver ON sessions(driver_id, started_at);
CREATE INDEX idx_metrics_session ON frame_metrics(session_id, timestamp);
```

---

## 8. RAG Voice Agent — Full Design

### 8.1 Architecture

```
DRIVER speaks →  [Microphone]
                     │
                     ▼
            ┌─────────────────────┐
            │  FASTER-WHISPER STT │  (local, base.en or small.en)
            │  Word-level timing  │  16kHz input → text
            │  VAD (silero)       │  Auto-detects speech end
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   INTENT ROUTER     │  (Ollama: phi4-mini or llama3.2:3b)
            │                     │
            │  Classes:           │
            │  · QUERY_HISTORY    │  "How did I do last week?"
            │  · QUERY_CURRENT    │  "What's my fatigue score now?"
            │  · COMMAND          │  "Take a break note"
            │  · COACHING         │  "Give me a tip"
            │  · CONVERSATION     │  "I'm feeling tired"
            │  · EMERGENCY        │  "I feel like I'm going to crash"
            └──────────┬──────────┘
                       │
           ┌───────────┼────────────┐
           │           │            │
           ▼           ▼            ▼
    ┌─────────┐ ┌──────────┐ ┌──────────────┐
    │ CURRENT │ │  RAG     │ │  COMMAND     │
    │ METRICS │ │ PIPELINE │ │  EXECUTOR    │
    │         │ │          │ │              │
    │ Direct  │ │ 1.Embed  │ │ Triggers     │
    │ from    │ │  query   │ │ agent tools  │
    │ working │ │ 2.Query  │ │ directly     │
    │ memory  │ │  ChromaDB│ │              │
    │         │ │ 3.Rank   │ │              │
    │         │ │  results │ │              │
    │         │ │ 4.Build  │ │              │
    │         │ │  prompt  │ │              │
    └────┬────┘ └────┬─────┘ └──────┬───────┘
         │           │               │
         └─────── MERGE ─────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │      OLLAMA GENERATOR    │
         │                          │
         │  Model: llama3.2:3b or   │
         │         phi4-mini        │
         │                          │
         │  System: "You are a      │
         │  driver safety assistant │
         │  for {driver_name}.      │
         │  Context from their      │
         │  driving history is      │
         │  provided. Be brief,     │
         │  direct, safety-focused."│
         │                          │
         │  Max tokens: 150 (voice!)│
         │  Temperature: 0.3        │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │     COQUI TTS ENGINE     │
         │                          │
         │  Model: tts_models/en/   │
         │  vctk/vits (multi-voice) │
         │                          │
         │  Fallback: pyttsx3       │
         │  Speed: 1.15x (alert)    │
         │         1.0x  (convers.) │
         └──────────┬───────────────┘
                    │
                    ▼
              [SPEAKER / HEADSET]
```

### 8.2 RAG Prompt Construction

```python
# voice_agent/rag_pipeline.py

RAG_SYSTEM_PROMPT = """
You are {driver_name}'s personal driver safety assistant.
You have direct access to their complete driving history.

CURRENT DRIVE STATUS:
  - Session duration: {session_minutes} min
  - Current fatigue score: {fatigue_score}/100 ({fatigue_level})
  - Current heart rate: {hr_bpm} BPM
  - Active alerts: {active_alerts}

RETRIEVED CONTEXT FROM DRIVER HISTORY:
{retrieved_context}

CONVERSATION HISTORY:
{conversation_history}

Instructions:
- Keep responses under 3 sentences (will be spoken aloud)
- If fatigue > 60, always mention safety first
- Reference specific past data when relevant ("Last Tuesday you also...")
- Never say "I don't know" — always provide best available guidance
- For emergencies, override everything and give pull-over instruction
"""

def build_rag_context(query: str, driver_id: str, top_k: int = 3) -> str:
    """
    1. Embed query with nomic-embed-text
    2. Search all ChromaDB collections with driver_id filter
    3. Rank by relevance + recency hybrid score
    4. Format as natural context paragraph
    """
```

### 8.3 Voice Interaction Examples

```
DRIVER: "How did I do today compared to yesterday?"
─────────────────────────────────────────────────
INTENT:  QUERY_HISTORY
RETRIEVE: last 2 sessions for driver_id
RESPONSE: "Today you're at a peak fatigue of 38, which is better than 
           yesterday's peak of 61. Your blink rate is steadier too. 
           Keep it up — you're on a good trend."

─────────────────────────────────────────────────
DRIVER: "I'm starting to feel tired"  
─────────────────────────────────────────────────
INTENT:  CONVERSATION → SAFETY
RETRIEVE: similar self-reported tiredness events
RESPONSE: "Noted. Your fatigue score just hit 52. Your history shows 
           you typically recover well with a 10-minute break. There's 
           a rest stop in 4 minutes — do you want me to log a break?"

─────────────────────────────────────────────────
DRIVER: "What's my worst time of day to drive?"
─────────────────────────────────────────────────
INTENT:  QUERY_HISTORY → PATTERN
RETRIEVE: pattern observations for this driver
RESPONSE: "Based on 47 of your sessions, your fatigue peaks between 
           2 and 4 PM. Your blink rate drops 30% and PERCLOS rises 
           most during those hours. Morning drives are your safest."
```

---

## 9. V3 Perception Upgrades

### 9.1 New Detectors

#### Saccade Velocity Tracker
```python
# perception/gaze_tracker.py — additions

class SaccadeAnalyzer:
    """
    Computes saccade velocity from successive iris positions.
    Drowsy saccades are slower (< 200°/s vs normal 400-600°/s).
    Also detects micro-saccades indicating sustained attention.
    """
    def analyze(self, iris_history: deque) -> dict:
        velocities = [dist(iris_history[i], iris_history[i-1]) * FPS
                      for i in range(1, len(iris_history))]
        return {
            'mean_velocity':   np.mean(velocities),
            'slow_saccade_pct': np.mean([v < SACCADE_SLOW_THRESH for v in velocities]),
            'fixation_duration': self._compute_fixation_duration(iris_history),
        }
```

#### Scene Context Analyzer
```python
# perception/scene_analyzer.py — NEW

class SceneAnalyzer:
    """
    Analyzes environment to contextually adjust detector weights.
    Uses frame ROI analysis — no external model needed.
    """
    def analyze(self, frame: np.ndarray) -> dict:
        return {
            'ambient_brightness': frame.mean(),     # adapt EAR threshold for dark
            'contrast':          frame.std(),       # low contrast = fog/rain
            'blue_channel_high': frame[:,:,0].mean() > 120,  # daytime sky
            'glare_detected':    self._detect_glare(frame),
            'lighting_class':    self._classify_lighting(frame),  # day/night/tunnel
        }

    def get_weight_adjustments(self, scene: dict) -> dict:
        """Returns per-signal weight multipliers based on scene."""
        if scene['lighting_class'] == 'night':
            return {'ear': 1.2, 'perclos': 1.3, 'rppg': 0.7}
        if scene['glare_detected']:
            return {'gaze': 0.6, 'ear': 1.1}
        return {}  # default weights
```

#### Micro-Expression Detector (Lightweight)
```python
# perception/ear_detector.py — V3 addition

class MicroExpressionDetector:
    """
    Detects rapid involuntary facial movements indicating:
    - Sudden alertness burst (post-microsleep)
    - Stress response
    - Confusion/disorientation
    Uses landmark velocity burst detection — no neural net needed.
    """
    def detect(self, landmark_history: deque) -> dict:
        velocities = self._compute_landmark_velocities(landmark_history)
        burst_score = max(velocities)
        return {
            'micro_expr_detected': burst_score > MICRO_EXPR_THRESH,
            'burst_score':         burst_score,
            'expression_type':     self._classify(velocities),  # alert/stress/confused
        }
```

### 9.2 Fatigue Score V3 — Prediction Module

```python
# perception/fatigue_score.py — V3 additions

class FatigueTrendPredictor:
    """
    Fits a linear regression on last 5 minutes of FatigueScore.
    Predicts value in 5 minutes.
    Triggers pre-emptive alerts BEFORE threshold crossing.
    """
    def __init__(self, window_sec: int = 300):
        self.history = deque(maxlen=window_sec)  # 1 sample/sec

    def add_sample(self, score: float, timestamp: float):
        self.history.append((timestamp, score))

    def predict_5min(self) -> dict:
        if len(self.history) < 60:  # Need 1 min of data
            return {'predicted': None, 'trend': 'insufficient_data'}
        
        times, scores = zip(*self.history)
        t_norm = np.array(times) - times[0]
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(t_norm.reshape(-1,1), scores)
        
        t_future = t_norm[-1] + 300  # 5 minutes ahead
        pred = reg.predict([[t_future]])[0]
        slope = reg.coef_[0]
        
        return {
            'predicted_5min': np.clip(pred, 0, 100),
            'slope_per_min':  slope * 60,  # score units per minute
            'trend':          'rising' if slope > 0.1 else 'falling' if slope < -0.1 else 'stable',
            'will_cross_warn': pred >= 45 and scores[-1] < 45,
            'will_cross_crit': pred >= 70 and scores[-1] < 70,
        }
```

---

## 10. Professional Dashboard — PyQt6 Design

### 10.1 Layout

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  DMS V3  [●REC] 01:23:45  Driver: Alex Johnson  Risk: MILD  [Settings] [Fleet] ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ┌─────────────────────────────────┐  ┌──────────────────────────────────────┐ ║
║  │                                 │  │  ANALYTICS PANEL                     │ ║
║  │      LIVE FEED PANEL            │  │                                      │ ║
║  │                                 │  │  FatigueScore ────────────────────── │ ║
║  │  [Camera feed with HUD overlay] │  │  100|     ╭─╮                        │ ║
║  │                                 │  │    0|─────╯ ╰────────── t           │ ║
║  │  ┌──────────┐  ┌─────────────┐  │  │                                      │ ║
║  │  │ Fatigue  │  │ Heart Rate  │  │  │  EAR Waveform ────────────────────── │ ║
║  │  │  Gauge   │  │   Gauge     │  │  │  0.40|╭──╮  ╭──╮  ╭──╮             │ ║
║  │  │  [○ 38] │  │  [♥ 72bpm] │  │  │  0.15|╯  ╰──╯  ╰──╯  ╰──────       │ ║
║  │  └──────────┘  └─────────────┘  │  │                                      │ ║
║  │                                 │  │  PERCLOS 60s ────────────────────── │ ║
║  │  ┌──────────┐  ┌─────────────┐  │  │  15%|        ╭──╮                   │ ║
║  │  │ Head     │  │  Attention  │  │  │   0%|────────╯  ╰──────────        │ ║
║  │  │ Compass  │  │  Heatmap    │  │  │                                      │ ║
║  │  └──────────┘  └─────────────┘  │  │  Head Yaw/Pitch ─────────────────── │ ║
║  │                                 │  │  +30|       ╭╮                       │ ║
║  │  ┌───────────────────────────┐  │  │  -30|───────╯╰───────────────       │ ║
║  │  │    ALERT STATUS BAR       │  │  │                                      │ ║
║  │  │  [🟡 MILD FATIGUE — 38]  │  │  │  HR BPM ───────────────────────────  │ ║
║  │  └───────────────────────────┘  │  │   80|────────────────────────       │ ║
║  └─────────────────────────────────┘  └──────────────────────────────────────┘ ║
║                                                                                 ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │  MEMORY & AGENT PANEL                                                    │  ║
║  │                                                                           │  ║
║  │  ┌──────────────────────────────┐  ┌──────────────────────────────────┐  │  ║
║  │  │  AGENT REASONING             │  │  VOICE AGENT CHAT                │  │  ║
║  │  │  ─────────────────────────── │  │  ─────────────────────────────── │  │  ║
║  │  │  [Safety Agent @ 01:22:30]   │  │  You: "How am I doing today?"    │  │  ║
║  │  │  Fatigue=38, trending +0.3/m │  │                                   │  │  ║
║  │  │  Retrieved 2 similar events  │  │  AI: "You're at 38/100 fatigue,  │  │  ║
║  │  │  from history. No action     │  │  better than your avg of 44.     │  │  ║
║  │  │  needed. Next check in 60s.  │  │  Keep going — you're solid."     │  │  ║
║  │  │                              │  │                                   │  │  ║
║  │  │  [Memory Agent @ 01:21:00]   │  │  You: "I'm feeling a bit tired"  │  │  ║
║  │  │  Stored 60s bulk metrics.    │  │                                   │  │  ║
║  │  │  Pattern: slow blink rising  │  │  AI: "Your score just hit 42.    │  │  ║
║  │  │  → embedded to ChromaDB.     │  │  Next exit is 8 min. Should I   │  │  ║
║  │  │                              │  │  log a break reminder?"          │  │  ║
║  │  └──────────────────────────────┘  └──────────────────────────────────┘  │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                 ║
║  ┌────────────────────────────────────────────────────────────────────────────┐ ║
║  │  SESSION TIMELINE  ─────────────────────────────────────────────────────  │ ║
║  │  00:00─────────────────────────────────────────────────────────── 01:23   │ ║
║  │         ▲ calibrated    ▲ phone alert   ▲ yawn x3   ▲ break ▲ now        │ ║
║  └────────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### 10.2 PyQt6 Panel Architecture

```python
# dashboard/pyqt/main_window.py  — Layout skeleton

class DMSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DMS V3 — Driver Monitoring System")
        self.setMinimumSize(1600, 900)

        # Central widget — 3-row splitter
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Row 1: Top bar (driver info, session time, risk indicator)
        main_layout.addWidget(TopBarWidget(), stretch=0)

        # Row 2: Main panels (2/3 video+gauges | 1/3 analytics charts)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(HUDPanel())           # Live feed + overlays
        top_splitter.addWidget(AnalyticsPanel())     # Real-time charts
        top_splitter.setSizes([2, 1])
        main_layout.addWidget(top_splitter, stretch=3)

        # Row 3: Memory + Agent panels
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(AgentReasoningPanel())
        bottom_splitter.addWidget(VoiceAgentChatPanel())
        main_layout.addWidget(bottom_splitter, stretch=1)

        # Row 4: Session timeline
        main_layout.addWidget(SessionTimelineWidget(), stretch=0)

        self.setCentralWidget(central)
        self._apply_dark_theme()
```

### 10.3 Color Theme & Design Language

```python
DMS_THEME = {
    # Background hierarchy
    'bg_primary':    '#0A0E1A',   # Near-black navy (main bg)
    'bg_secondary':  '#111827',   # Panel bg
    'bg_card':       '#1C2333',   # Card bg
    'bg_elevated':   '#232D42',   # Hover/selected

    # Status colors
    'status_normal':   '#10B981',  # Emerald — safe
    'status_mild':     '#F59E0B',  # Amber — attention
    'status_warning':  '#EF4444',  # Red — warning
    'status_critical': '#DC2626',  # Deep red — critical
    'status_info':     '#3B82F6',  # Blue — information

    # Text
    'text_primary':   '#F1F5F9',   # Off-white
    'text_secondary': '#94A3B8',   # Slate
    'text_muted':     '#475569',   # Dark slate

    # Accent
    'accent_blue':   '#3B82F6',
    'accent_purple': '#8B5CF6',

    # Gauge fill colors (fatigue score gradient)
    'gauge_0':  '#10B981',   # 0% — green
    'gauge_45': '#F59E0B',   # 45% — amber
    'gauge_70': '#EF4444',   # 70% — red
    'gauge_90': '#DC2626',   # 90% — critical red

    # Font
    'font_family':  'Inter',     # Primary
    'font_mono':    'JetBrains Mono',  # Data/metrics
    'font_size_lg': 28,
    'font_size_md': 14,
    'font_size_sm': 11,
}
```

---

## 11. FastAPI Web Interface

### 11.1 REST API Endpoints

```python
# dashboard/web/api_server.py

# ── Session endpoints ──
GET  /api/sessions                     # List all sessions
GET  /api/sessions/{session_id}        # Session details + metrics
GET  /api/sessions/{session_id}/report # AI-generated report
GET  /api/sessions/{session_id}/events # Event timeline

# ── Driver endpoints ──
GET  /api/drivers                      # List all drivers
GET  /api/drivers/{driver_id}          # Driver profile + stats
GET  /api/drivers/{driver_id}/patterns # Detected behavioral patterns
POST /api/drivers/{driver_id}/note     # Add coaching note

# ── Live endpoints ──
GET  /api/live/metrics                 # Current frame metrics
GET  /api/live/alerts                  # Active alerts
POST /api/live/dismiss/{alert_id}      # Dismiss an alert

# ── Memory / RAG endpoints ──
POST /api/memory/query                 # RAG query (body: {query, driver_id})
GET  /api/memory/insights/{driver_id}  # All stored insights

# ── Fleet endpoints ──
GET  /api/fleet/overview               # All active sessions + risk summary
GET  /api/fleet/risk-report            # Drivers by risk tier

# ── WebSocket ──
WS   /ws/live                          # 1Hz live metrics stream (JSON)
WS   /ws/alerts                        # Real-time alert events
```

### 11.2 WebSocket Live Feed

```python
# dashboard/web/api_server.py  — WebSocket handler

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            metrics = working_memory.get_current_metrics()
            payload = {
                'timestamp':     time.time(),
                'fatigue_score': metrics['fatigue_score'],
                'fatigue_level': metrics['fatigue_level'],
                'ear':           metrics['ear'],
                'perclos':       metrics['perclos'],
                'hr_bpm':        metrics['hr_bpm'],
                'active_alerts': metrics['active_alerts'],
                'agent_status':  metrics['agent_status'],
                'prediction':    metrics['fatigue_prediction'],
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
```

---

## 12. Ollama Model Strategy

### 12.1 Model Selection Guide

```
Task                        Recommended Model       Fallback
──────────────────────────────────────────────────────────────────────
Agentic reasoning           llama3.2:3b             phi4-mini
Intent classification       phi4-mini               llama3.2:1b
RAG generation (voice)      llama3.2:3b             phi3.5
Session report              llama3.1:8b             llama3.2:3b
Embedding (ChromaDB)        nomic-embed-text        mxbai-embed-large
Safety coaching             llama3.2:3b             qwen2.5:3b
Fleet analysis              llama3.1:8b             llama3.2:3b

GPU VRAM Guide:
  4GB  → llama3.2:1b  (limited but functional)
  8GB  → llama3.2:3b  (recommended baseline)
  16GB → llama3.1:8b  (full capability)
  CPU  → phi4-mini    (fast enough at 4 cores)
```

### 12.2 Auto Model Detection

```python
# config.py  — Model auto-selection

def detect_best_ollama_models() -> dict:
    """Query Ollama and select best available models for each role."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        available = [m['name'] for m in resp.json().get('models', [])]
    except:
        return FALLBACK_MODEL_MAP

    def pick(preference_list: list) -> str:
        for model in preference_list:
            if any(model.split(':')[0] in a for a in available):
                return model
        return preference_list[-1]  # Last = most compatible fallback

    return {
        'agent':     pick(['llama3.1:8b', 'llama3.2:3b', 'phi4-mini', 'llama3.2:1b']),
        'intent':    pick(['phi4-mini', 'llama3.2:1b', 'qwen2.5:1.5b']),
        'voice_rag': pick(['llama3.2:3b', 'phi4-mini', 'llama3.2:1b']),
        'report':    pick(['llama3.1:8b', 'llama3.2:3b', 'phi4-mini']),
        'embed':     pick(['nomic-embed-text', 'mxbai-embed-large']),
    }
```

---

## 13. Config V3 — Complete

```python
# config.py  — DMS V3 unified configuration

import os
from typing import Optional

# ─── System ──────────────────────────────────────────────────────────────────
VERSION              = "3.0.0"
DEBUG_MODE           = False
LOG_LEVEL            = "INFO"

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
TARGET_FPS           = 30
FLIP_HORIZONTAL      = True

# ─── MediaPipe ───────────────────────────────────────────────────────────────
MAX_NUM_FACES        = 1
REFINE_LANDMARKS     = True
MIN_DETECTION_CONF   = 0.7
MIN_TRACKING_CONF    = 0.7

# ─── Kalman Filter ───────────────────────────────────────────────────────────
KALMAN_ENABLED            = True
KALMAN_PROCESS_NOISE      = 0.005
KALMAN_MEASUREMENT_NOISE  = 0.05

# ─── Calibration ─────────────────────────────────────────────────────────────
CALIBRATION_DURATION_SEC  = 30
CALIBRATION_PROFILE_DIR   = "data/driver_profiles"
DEFAULT_DRIVER_ID         = "default"
AUTO_CALIBRATE_ON_START   = True
CONTINUOUS_RECALIBRATE    = True   # V3: background drift correction

# ─── EAR V3 ──────────────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.25
EAR_CONSEC_FRAMES    = 48
EAR_WARN_FRAMES      = 20
BLINK_VELOCITY_SLOW  = -0.012
BLINK_SLOW_THRESH    = 0.35
BLINK_RATE_MIN       = 10
BLINK_RATE_MAX       = 25
MICRO_EXPR_ENABLED   = True          # V3 NEW
MICRO_EXPR_THRESH    = 0.045

LEFT_EYE_INDICES     = [362,385,387,263,373,380]
RIGHT_EYE_INDICES    = [33,160,158,133,153,144]

# ─── MAR V3 ──────────────────────────────────────────────────────────────────
MAR_THRESHOLD        = 0.60
MAR_CONSEC_FRAMES    = 20
YAWN_RATE_ALERT      = 3

# ─── Head Pose V3 ────────────────────────────────────────────────────────────
PITCH_DOWN_THRESH    = 20
PITCH_UP_THRESH      = -15
YAW_THRESH           = 30
ROLL_THRESH          = 25
HEAD_SWAY_WINDOW     = 90
HEAD_JERK_THRESH     = 15.0

# ─── Gaze V3 ─────────────────────────────────────────────────────────────────
GAZE_THRESH          = 0.35
GAZE_CONSEC_FRAMES   = 60
FIXATION_MIN_FRAMES  = 15
HEATMAP_DECAY        = 0.995
SACCADE_SLOW_THRESH  = 200.0         # V3 NEW: °/s below = drowsy saccade

# ─── PERCLOS V3 ──────────────────────────────────────────────────────────────
PERCLOS_WINDOW_SEC   = 60
PERCLOS_ALERT_THRESH = 0.15
PERCLOS_WARN_THRESH  = 0.08
PERCLOS_CLOSURE_RATIO = 0.70
PERCLOS_PREDICTION   = True          # V3 NEW

# ─── rPPG V3 ─────────────────────────────────────────────────────────────────
RPPG_ENABLED         = True
RPPG_WINDOW_SEC      = 10
RPPG_HR_MIN          = 42
RPPG_HR_MAX          = 180
RPPG_HRV_ENABLED     = True          # V3 NEW
RPPG_STRESS_THRESH   = 95
RPPG_LOW_THRESH      = 55

# ─── Scene Analyzer V3 ───────────────────────────────────────────────────────
SCENE_ANALYZER_ENABLED = True        # V3 NEW
SCENE_SAMPLE_RATE_SEC  = 5.0         # Analyze scene every 5s

# ─── Fatigue Score V3 ────────────────────────────────────────────────────────
FATIGUE_NORMAL       = 25
FATIGUE_MILD         = 45
FATIGUE_WARNING      = 70
FATIGUE_CRITICAL     = 85
FATIGUE_PREDICTION_ENABLED = True    # V3 NEW
FATIGUE_PREDICTION_WINDOW  = 300     # 5 min history for trend

# Signal weights (total must = 1.0)
FATIGUE_WEIGHTS = {
    'ear':       0.22,
    'perclos':   0.18,
    'blink':     0.13,
    'head_sway': 0.09,
    'mar':       0.09,
    'gaze':      0.10,
    'rppg':      0.10,
    'scene':     0.05,
    'micro_expr':0.04,
}

# ─── Object Detection ────────────────────────────────────────────────────────
YOLO_ENABLED         = True
YOLO_MODEL           = "yolov8n.pt"
YOLO_CONF            = 0.45
YOLO_CLASSES         = [67, 41, 39, 73]  # phone, cup, bottle, laptop
YOLO_INTERVAL_FRAMES = 10

# ─── Alert System ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC   = 8.0
ALERT_ESCALATION_STEPS = 3
TTS_ENABLED          = True
TTS_RATE             = 175
TTS_VOLUME           = 0.9
TTS_ENGINE           = "coqui"       # coqui / pyttsx3
COQUI_MODEL          = "tts_models/en/vctk/vits"
COQUI_SPEAKER        = "p267"

# ─── Memory System ───────────────────────────────────────────────────────────
MEMORY_DB_PATH       = "data/sessions.db"
CHROMA_DB_PATH       = "data/chroma_db"
CHROMA_COLLECTION_PREFIX = "dms_v3"
METRICS_SAMPLE_RATE_SEC  = 1.0       # Log metrics every 1s (not 30fps)
BULK_EMBED_INTERVAL_SEC  = 60.0      # Embed session chunks every 60s

# ─── Agentic System ──────────────────────────────────────────────────────────
AGENT_ENABLED        = True
AGENT_CHECK_INTERVAL = 1.0           # Agent reasoning cycle (seconds)
AGENT_REASONING_LOG  = True          # Show agent chain-of-thought in UI

# ─── Voice Agent ─────────────────────────────────────────────────────────────
VOICE_AGENT_ENABLED  = True
WHISPER_MODEL        = "base.en"     # base.en / small.en / medium.en
WHISPER_DEVICE       = "cpu"         # cpu / cuda
WHISPER_VAD          = True          # Voice activity detection
VOICE_CONTEXT_TURNS  = 5             # Conversation history length
VOICE_MAX_TOKENS     = 150           # Keep responses brief (spoken)

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_HOST          = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT       = 30
OLLAMA_REPORT_INTERVAL_MIN = 5
OLLAMA_AUTO_DETECT   = True          # Auto-select best available model
# These will be overridden by auto-detection if OLLAMA_AUTO_DETECT=True:
OLLAMA_AGENT_MODEL   = "llama3.2:3b"
OLLAMA_INTENT_MODEL  = "phi4-mini"
OLLAMA_VOICE_MODEL   = "llama3.2:3b"
OLLAMA_REPORT_MODEL  = "llama3.1:8b"
OLLAMA_EMBED_MODEL   = "nomic-embed-text"

# ─── Web Interface ───────────────────────────────────────────────────────────
WEB_ENABLED          = True
WEB_HOST             = "0.0.0.0"
WEB_PORT             = 8080
WEB_RELOAD           = False

# ─── Dashboard ───────────────────────────────────────────────────────────────
DASHBOARD_MODE       = "pyqt"        # pyqt / headless / web-only
DASHBOARD_THEME      = "dark"
ANALYTICS_HISTORY_MIN = 5            # Minutes of history shown in charts
```

---

## 14. Threading Architecture V3

```
Thread Name          Priority  Affinity     Purpose
────────────────────────────────────────────────────────────────────────────────
MainThread           REALTIME  Core 0       Camera capture + MediaPipe + Kalman
                               Core 1       + All detectors + FatigueScore
                                            + Overlay rendering
                                            Target: <33ms per frame (30fps)

YOLOThread           NORMAL    Core 2       Background YOLOv8 inference
                                            Posts results to shared dict
                                            ~15ms per inference, non-blocking

rPPGThread           NORMAL    Core 2       FFT on accumulated frames (30fps)
                                            Runs full FFT every 30 frames

AgentThread          LOW       Core 3       LangGraph orchestration cycle
                                            Runs every 1s, calls Ollama async
                                            Non-blocking to main thread

MemoryThread         LOW       Core 3       Bulk SQLite writes every 1s
                                            ChromaDB embedding every 60s

VoiceAgentThread     NORMAL    Core 2       Whisper STT → intent → RAG → TTS
                                            Runs on wake word or push-to-talk

TTSAlertThread       HIGH      Core 2       Alert TTS synthesis + playback
                                            Priority queue, preempts voice agent

WebServerThread      LOW       Core 3       FastAPI + WebSocket server
                                            asyncio event loop

SceneAnalyzerThread  LOW       Core 3       Ambient scene analysis every 5s

Auto-Calibrate       LOW       Core 3       Background threshold drift correction
                                            Runs every 5 minutes in background

────────────────────────────────────────────────────────────────────────────────
Frame latency budget (30fps = 33ms):
  Camera capture:         ~3ms
  MediaPipe + Kalman:     ~9ms
  EAR + blink + micro:    ~2ms
  MAR:                    ~1ms
  Head pose V3:           ~3ms
  Gaze + saccade:         ~2ms
  PERCLOS + prediction:   ~2ms
  Fatigue Score V3:       ~2ms
  Overlay V3:             ~7ms
  ─────────────────────────────
  Total:                  ~31ms  ✅ Within 33ms budget
  
  (YOLO, rPPG, agents, memory, voice = all background threads)
```

---

## 15. Setup & Installation V3

```bash
#!/bin/bash
# setup.sh — DMS V3 Full Installation

set -e
echo "=== DMS V3 Setup ==="

# 1. Python environment
python3 -m venv venv
source venv/bin/activate

# 2. Core perception stack
pip install mediapipe opencv-python numpy scipy scikit-learn pillow

# 3. Detection
pip install ultralytics  # YOLOv8

# 4. Audio + TTS
pip install pyttsx3 pyaudio
pip install TTS  # Coqui TTS (larger download ~1GB)
# OR lightweight fallback: pip install pyttsx3

# 5. Whisper STT
pip install faster-whisper
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu')"

# 6. Memory stack
pip install chromadb
pip install sqlalchemy

# 7. Agentic layer
pip install langgraph langchain-core langchain-community

# 8. Web interface
pip install fastapi uvicorn websockets pydantic

# 9. Dashboard
pip install PyQt6 pyqtgraph

# 10. Utilities
pip install requests python-dotenv loguru rich

# 11. Ollama check + model pull
echo "=== Checking Ollama ==="
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama running. Pulling required models..."
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    # Optional but recommended:
    # ollama pull phi4-mini
    # ollama pull llama3.1:8b
else
    echo "WARNING: Ollama not running. Start with: ollama serve"
    echo "Then run: ollama pull llama3.2:3b && ollama pull nomic-embed-text"
fi

# 12. Create data directories
mkdir -p data/chroma_db data/driver_profiles data/models/whisper data/reports

echo "=== DMS V3 Setup Complete ==="
echo "Run: python main.py"
echo "Web UI: http://localhost:8080"
```

### Requirements V3

```
# requirements.txt

# Perception
mediapipe>=0.10.14
opencv-python>=4.10.0
numpy>=1.26.0
scipy>=1.13.0
scikit-learn>=1.5.0
Pillow>=10.0.0
ultralytics>=8.2.0

# Voice Agent
faster-whisper>=1.0.3
TTS>=0.22.0        # Coqui TTS
pyttsx3>=2.90
pyaudio>=0.2.14    # microphone input
silero-vad>=5.1    # voice activity detection

# Memory
chromadb>=0.5.0
sqlalchemy>=2.0.0

# Agentic
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-community>=0.3.0

# Web Interface
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
websockets>=13.1
pydantic>=2.9.0

# Dashboard
PyQt6>=6.7.0
pyqtgraph>=0.13.7

# Utilities
requests>=2.32.0
python-dotenv>=1.0.1
loguru>=0.7.2
rich>=13.8.0
```

---

## 16. Key Improvements Table — V1 → V2 → V3

| Feature | V1 | V2 | V3 |
|---|---|---|---|
| **Landmark smoothing** | 5-frame avg | Kalman per-landmark | Kalman + drift correction |
| **Thresholds** | Fixed | 30s calibration | Continuous auto-tune |
| **Heart rate** | None | rPPG green FFT | + HRV + stress coherence |
| **Fatigue signals** | 5 | 7 | **9** (+ micro-expr, saccade, scene) |
| **Fatigue prediction** | None | None | **5-min linear regression** |
| **AI model** | 5-min report | + Contextual mid-drive | **Agentic reasoning (LangGraph)** |
| **AI memory** | None | Profile JSON | **ChromaDB + SQLite + working** |
| **AI reasoning** | None | None | **Chain-of-thought logged** |
| **Voice** | TTS alerts | TTS escalation | **Full bidirectional RAG voice agent** |
| **STT** | None | None | **Whisper (local, offline)** |
| **TTS quality** | pyttsx3 | pyttsx3 priority queue | **Coqui neural TTS** |
| **RAG** | None | None | **ChromaDB + nomic-embed-text** |
| **Scene context** | None | None | **Lighting/glare/day-night adaptation** |
| **UI framework** | OpenCV | OpenCV + matplotlib | **PyQt6 professional dashboard** |
| **Web interface** | None | None | **FastAPI + WebSocket** |
| **Fleet support** | None | None | **Multi-driver, fleet overview** |
| **Session replay** | None | None | **Timeline + playback** |
| **Coaching** | None | None | **Personalized improvement tracking** |
| **Pattern detection** | None | None | **Behavioral pattern learning** |
| **Prediction** | None | None | **Pre-emptive alerts before threshold** |
| **API** | None | None | **Full REST + WebSocket** |
| **Testing** | None | None | **pytest suite + fixtures** |

---

## 17. Implementation Roadmap

### Phase 1 — Core V3 (Week 1-2)
```
Priority: Get enhanced perception + memory running
─────────────────────────────────────────────────
□ Port all V2 detectors, add saccade + micro-expression + scene analyzer
□ Implement FatigueScore V3 with 9 signals + prediction module
□ Build Memory V3: working memory + SQLite schema + ChromaDB setup
□ Implement continuous auto-calibration
□ Basic PyQt6 HUD panel (replaces OpenCV window)
□ Embed session insights into ChromaDB at session end
```

### Phase 2 — Agentic Layer (Week 2-3)
```
Priority: Safety Agent + Memory Agent operational
─────────────────────────────────────────────────
□ Build Orchestrator with LangGraph state graph
□ Implement Safety Agent with tools: trigger_alert, voice_warn
□ Implement Memory Agent: save_event, query_history, embed_insight
□ Implement Prediction Agent: forecast_fatigue, preemptive_alert
□ Wire agent reasoning output to PyQt6 panel
□ Test agent behavior with simulated metric streams
```

### Phase 3 — Voice Agent (Week 3-4)
```
Priority: Driver can talk to the system
─────────────────────────────────────────────────
□ Integrate faster-whisper with VAD (silero)
□ Build intent router (Ollama phi4-mini)
□ Build RAG pipeline (ChromaDB → Ollama → Coqui TTS)
□ Implement conversation context manager (5-turn)
□ Wire voice agent to working memory (current metrics)
□ Add push-to-talk button in PyQt6 + wake-word detection
□ Test all intent classes with real scenarios
```

### Phase 4 — Full Dashboard + Web (Week 4-5)
```
Priority: Professional UI + fleet capability
─────────────────────────────────────────────────
□ Complete PyQt6: analytics panel, memory browser, session timeline
□ Build FastAPI server: all REST endpoints + WebSocket streams
□ Implement Report Agent + Coaching Agent
□ Add session replay in PyQt6
□ Build fleet overview page
□ Write pytest suite (perception, memory, voice, agents)
□ Performance profiling — ensure <33ms frame budget maintained
```

### Phase 5 — Polish & Testing (Week 5-6)
```
Priority: Production readiness
─────────────────────────────────────────────────
□ Real-world testing: different drivers, lighting, cameras
□ Tune adaptive thresholds for glasses, beard, lighting edge cases
□ rPPG calibration for different skin tones
□ Voice agent accuracy testing (all intent classes)
□ Memory retrieval relevance testing
□ Load testing: 8h session with memory growth
□ Documentation: API.md, DEPLOYMENT.md, TUNING.md
□ Docker compose for easy deployment
```

---

## 18. Testing Strategy

### 18.1 Test Structure

```python
# tests/test_perception.py
def test_ear_kalman_smoothing():
    """Verify Kalman reduces jitter by >50% vs raw EAR"""

def test_fatigue_score_weights():
    """Verify weights sum to 1.0 and score stays 0-100"""

def test_fatigue_prediction_accuracy():
    """With synthetic rising fatigue data, predict within ±5 of actual"""

def test_scene_analyzer_night_detection():
    """Dark frame correctly classified as 'night', weights adjusted"""

# tests/test_memory.py
def test_working_memory_thread_safety():
    """Concurrent reads/writes do not corrupt state"""

def test_episodic_memory_session_crud():
    """Create, read, update session in SQLite"""

def test_chroma_similarity_search():
    """Query retrieves semantically similar past event"""

# tests/test_voice_agent.py
def test_intent_router_query_history():
    """'How did I do last week?' → QUERY_HISTORY intent"""

def test_rag_pipeline_grounding():
    """Response contains driver-specific data, not generic advice"""

def test_voice_response_length():
    """All voice responses ≤ 3 sentences (spoken brevity)"""

# tests/test_agents.py
def test_safety_agent_critical_trigger():
    """fatigue_score=75 → Safety Agent triggered, alert fired"""

def test_prediction_agent_preemptive():
    """Rising trend → Pre-emptive alert before crossing 45"""

def test_orchestrator_routing():
    """Score=80 → safety_agent; Score=30 → memory_agent"""
```

---

## 19. Production Deployment Checklist

```
Hardware:
  □ Laptop/PC with webcam (720p minimum, 1080p recommended)
  □ 8GB RAM minimum (16GB recommended for llama3.1:8b)
  □ 4 CPU cores minimum (agentic threads need isolation)
  □ 50GB disk (models + ChromaDB growth)
  □ Microphone (USB or built-in) for voice agent
  □ Speakers/headset for TTS output

Software:
  □ Python 3.11+
  □ Ollama installed + running (ollama serve)
  □ Models pulled: llama3.2:3b + nomic-embed-text
  □ All requirements installed (pip install -r requirements.txt)
  □ Data directories created (setup.sh)

Pre-run verification:
  □ ollama list  → shows required models
  □ python -c "import chromadb, langgraph, faster_whisper, TTS" → no errors
  □ python -c "import PyQt6, pyqtgraph" → no errors
  □ Camera accessible: python -c "import cv2; c=cv2.VideoCapture(0); print(c.read()[0])"
  □ Microphone accessible: python -c "import pyaudio; p=pyaudio.PyAudio(); print(p.get_device_count())"

First run:
  □ Run calibration for each new driver (30s)
  □ Verify FatigueScore responds to simulated eye closure
  □ Test voice: say "How am I doing?" → receive spoken response
  □ Confirm ChromaDB populating after 60s
  □ Open http://localhost:8080 → web interface loads
```

---

## 20. What Makes This V3 Fundamentally Different

```
V1/V2 = "Alarm System"         V3 = "Safety Co-Pilot"
────────────────────────────────────────────────────────────────────────
Waits for threshold crossing → Pre-empts risk before it peaks
Forgets between sessions     → Remembers and learns from every drive
Fires generic alerts         → Personalizes to each driver's baseline
Reports after the fact       → Reasons in real-time (1s cycle)
Driver is passive            → Driver can ask questions, get answers
Monitors driver              → Partners with driver
Session-scoped knowledge     → Lifetime memory and pattern recognition
Rule-based decisions         → Agent reasoning + reflection
TTS barks                    → Full conversation with context
One dashboard                → PyQt6 + Web + API for fleet
```

---

*DMS V3 — Agentic Driver Safety Co-Pilot*  
*Stack: MediaPipe · Kalman · rPPG · YOLOv8 · LangGraph · ChromaDB · Whisper · Coqui TTS · Ollama · PyQt6 · FastAPI*  
*Cost: $0 · 100% Offline · No Cloud · No API Keys · Runs on Any Laptop*
