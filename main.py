import threading
import uvicorn
import time
import random
from dashboard.web.api_server import app, live_state
from dashboard.pyqt.main_window import run_gui
from agents.orchestrator import create_orchestrator
from memory.memory_manager import MemoryManager
from voice_agent.voice_agent import VoiceAgent

def main_perception_loop():
    """
    Main loop for V3 DMS.
    Reads frames, runs detectors, updates WorkingMemory, 
    triggers Orchestrator, and handles Voice Agent.
    """
    import cv2
    from perception.face_detector import FaceDetector
    from perception.ear_detector import EARDetector
    from perception.fatigue_score import FatigueScoreEngine

    driver_id = "drv_8891"
    
    # Initialize Memory & Agents
    memory = MemoryManager(driver_id)
    orchestrator = create_orchestrator()
    voice = VoiceAgent(memory.semantic)
    
    print("[SYSTEM] Starting DMS V3 with Real Camera...")
    
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    ear_detector = EARDetector()
    fatigue_engine = FatigueScoreEngine()
    
    turn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame = cv2.flip(frame, 1)
        # We need a copy of the frame to pass to the UI safely
        live_state["frame"] = frame.copy()

        # Run FaceDetector
        has_face = face_detector.process(frame)
        current_score = live_state.get("fatigue_score", 0.0)
        lvl = "normal"
        
        if has_face:
            # We draw the mesh for the UI
            frame = face_detector.draw_mesh(frame)
            live_state["frame"] = frame.copy()
            
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            left_pts = face_detector.get_pixel_coords_batch(left_eye_indices)
            right_pts = face_detector.get_pixel_coords_batch(right_eye_indices)
            
            if left_pts is not None and right_pts is not None:
                # Update EAR
                smoothed_ear, state, is_alert = ear_detector.update(left_pts, right_pts)
                
                # Update Fatigue Score
                current_score, lvl = fatigue_engine.update(
                    ear_components=ear_detector.fatigue_components(),
                    perclos_components={},
                    head_components={},
                    mar_components={},
                    gaze_components={},
                    rppg_components={}
                )
        else:
            # If no face, score naturally decays or stays?
            pass
        

        live_state["fatigue_score"] = current_score
        live_state["fatigue_level"] = lvl
        
        # Create orchestrator state
        agent_state = {
            "driver_id": driver_id,
            "session_id": memory.session_id,
            "metrics": live_state,
            "fatigue_score": current_score,
            "fatigue_level": lvl,
            "active_alerts": [],
            "last_agent": "none",
            "reasoning": "",
            "memory_context": "",
            "action_taken": "",
            "turn": turn
        }
        
        # Trigger orchestrator (conditionally or on interval)
        if turn % 5 == 0:  # e.g., run agents every 5 ticks
            try:
                # The state graph processes the state and returns the final state
                final_state = orchestrator.invoke(agent_state)
                live_state["agent_status"] = f"[{final_state.get('last_agent', 'none')}] {final_state.get('action_taken', '')}"
            except Exception as e:
                import traceback
                print(f"[ORCHESTRATOR ERROR] {e}")
                traceback.print_exc()
                
        turn += 1

if __name__ == "__main__":
    # 1. Start core perception & agent loop
    t1 = threading.Thread(target=main_perception_loop, daemon=True)
    t1.start()
    
    # 2. Start FastAPI REST & WebSocket server
    def run_api():
        import logging
        logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="critical")
        
    t2 = threading.Thread(target=run_api, daemon=True)
    t2.start()
    
    # 3. Start PyQt6 UI (Blocking, must be main thread)
    print("Starting UI on main thread...")
    run_gui(live_state)
