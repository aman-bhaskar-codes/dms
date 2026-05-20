"""
DMS V4 Main Entry Point.
Coordinates Threaded Web Dashboard, OpenCV Camera Loop, and AI Agents.
"""
import asyncio
import cv2
import threading
import time
import webbrowser
from config import settings
from dashboard.web_dashboard import start_server, app as fastapi_app, update_state_safe, read_state_safe
from dashboard.overlay import HUDOverlay
from detectors.face_detector import FaceDetector
from detectors.ear_detector import EARDetector
from detectors.mar_detector import MARDetector
from detectors.head_pose import HeadPoseDetector
from detectors.gaze import GazeTracker
from detectors.rppg import RPPGEstimator
from detectors.perclos import PERCLOSEngine
from detectors.fatigue_score import FatigueScoreEngine
from calibration.driver_calibration import DriverCalibration
from memory.database import DatabaseManager
from memory.driver_memory import DriverMemory
from ai.ollama_client import OllamaClient
from voice_agent.cognition import VoiceCognitionAgent
from observability.tracer import frame_timer, signal_auditor, alert_auditor
from alerts.tts_engine import TTSEngine
from alerts.telegram_notifier import TelegramNotifier
from alerts.alert_engine import AlertEngine


async def async_main():
    print("🚗 Starting DMS V4 (Ollama Edition)...")
    
    # 1. Initialize DB and Memory
    db = DatabaseManager()
    await db.initialize()
    memory = DriverMemory(db, driver_id="drv_v4_test")
    await memory.start_session()
    
    # 2. Initialize AI & Alerts
    ollama_client = OllamaClient()
    tts = TTSEngine()
    tts.start()
    telegram = TelegramNotifier()
    alert_engine = AlertEngine(tts, memory, telegram)
    
    # 3. Start FastAPI Dashboard in a background thread
    print("[SYSTEM] Starting Web Dashboard on http://localhost:8080")
    web_thread = threading.Thread(target=start_server, daemon=True)
    web_thread.start()
    
    # Give web server a moment to bind
    await asyncio.sleep(1)
    
    # Auto-open dashboard in default web browser
    webbrowser.open("http://localhost:8080")
    
    # 4. Initialize Perception Pipeline
    print("[SYSTEM] Initializing Camera and Perception Models...")
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        print(f"❌ Failed to open camera {settings.camera_index}")
        return

    face_detector = FaceDetector()
    ear_detector = EARDetector()
    mar_detector = MARDetector()
    head_pose = HeadPoseDetector()
    gaze_tracker = GazeTracker()
    rppg_estimator = RPPGEstimator()
    perclos = PERCLOSEngine()
    fatigue_engine = FatigueScoreEngine()
    voice_agent = VoiceCognitionAgent(
        get_metrics=lambda: read_state_safe()[0],
        get_memory_context=memory.get_session_context,
        ollama_client=ollama_client,
        tts_engine=tts
    )
    voice_agent.start()
    hud = HUDOverlay()
    
    # Optional YOLO
    object_detector = None
    if settings.yolo_enabled:
        from detectors.object_detector import ObjectDetector
        object_detector = ObjectDetector()
        object_detector.start()
        
    calibration = DriverCalibration("drv_v4_test")
    if settings.auto_calibrate_on_start:
        calibration.start_calibration()
        tts.speak("Starting calibration. Please look forward and blink naturally.", priority=2)

    # State loop variables
    frame_count = 0
    last_ai_tip_time = time.time()
    
    try:
        while True:
            # Check for remote calibration trigger from the dashboard
            if getattr(fastapi_app.state, "trigger_calibrate", False):
                fastapi_app.state.trigger_calibrate = False
                calibration.start_calibration()
                tts.speak("Starting manual calibration. Please look forward and blink naturally.", priority=2)

            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
                
            if settings.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # --- PERCEPTION ---
            metrics = {}
            if object_detector:
                object_detector.submit_frame(frame)
                metrics["detections"] = object_detector.get_detections()
                
            has_face = face_detector.process(frame)
            if has_face:
                # Mesh drawing for HUD
                frame = face_detector.draw_mesh(frame)
                
                with frame_timer.measure("perception"):
                    # EAR
                    ear_data = ear_detector.update(face_detector, calibration)
                    perclos_data = perclos.update(ear_data["ear"])
                    
                    # MAR
                    mar_data = mar_detector.update(face_detector)
                    
                    # Head
                    head_data = head_pose.update(face_detector, frame)
                    
                    # Gaze
                    gaze_data = gaze_tracker.update(face_detector)
                    
                    # rPPG
                    fh_pts = face_detector.get_pixel_coords_batch(getattr(settings, "FOREHEAD_PTS", [10, 338, 297, 332, 284, 109, 67, 103, 54]))
                    rppg_data = rppg_estimator.update(frame, fh_pts)
                
                # Calibration feed
                if calibration._collecting:
                    if calibration.feed(ear_data["ear"], mar_data["mar"]):
                        ear_detector.set_threshold(calibration.ear_threshold)
                        mar_detector.set_threshold(calibration.mar_threshold)
                        perclos.set_baseline(calibration.profile.ear_open_baseline)
                        tts.speak("Calibration complete. Drive safely.", priority=2)
                        
                # Fatigue Fusion
                fatigue_data = fatigue_engine.update(
                    ear_data=ear_data,
                    perclos_data=perclos_data,
                    mar_data=mar_data,
                    head_data=head_data,
                    gaze_data=gaze_data,
                    rppg_data=rppg_data
                )
                
                # Audit
                signal_auditor.record({"gaze": gaze_data.get("gaze_x", 0), "rppg": rppg_data.get("hr_bpm", 0), "perclos": perclos_data["perclos"]})
                
                metrics.update(ear_data)
                metrics.update(perclos_data)
                metrics.update(mar_data)
                metrics.update(head_data)
                metrics.update(gaze_data)
                metrics.update(rppg_data)
                metrics["latency_ms"] = frame_timer.get_stats().get("perception", {}).get("mean_ms", 0.0)
                
                metrics["fatigue_score"] = fatigue_data["score"]
                metrics["fatigue_level"] = fatigue_data["level"]
                metrics["prediction_3min"] = fatigue_data["prediction_3min"]
                
                memory.update_fatigue(fatigue_data["score"])
                
                # --- ALERTS ---
                if fatigue_data["level"] in ["critical", "warning"]:
                    await alert_engine.trigger("fatigue", fatigue_data["level"], fatigue_data["score"])
                
                if ear_data["state"] == "critical":
                    await alert_engine.trigger("microsleep", "critical", fatigue_data["score"])
                    
                if head_data["state"] == "distracted":
                    await alert_engine.trigger("distraction", "warning", fatigue_data["score"])
                    
            # --- DASHBOARD & OVERLAY ---
            hud_frame = hud.render(frame, metrics)
            
            # Send to FastAPI global state thread-safely
            update_state_safe(metrics, hud_frame)
            
            # (Optional) local CV2 show if debugging
            if settings.DEBUG_MODE:
                cv2.imshow("DMS V4 Debug HUD", hud_frame)
                if cv2.waitKey(1) & 0xFF == settings.QUIT_KEY:
                    break

            # --- AI COACHING (Every 5 mins) ---
            now = time.time()
            if now - last_ai_tip_time > (settings.ollama_report_interval_min * 60):
                last_ai_tip_time = now
                ctx = await memory.get_session_context(metrics)
                # Run AI tip in background task so it doesn't block camera
                async def fetch_and_speak_tip():
                    tip = await ollama_client.get_driving_tip(ctx, metrics)
                    print(f"[Ollama Tip] {tip}")
                    await memory.log_ai_tip(ctx, tip)
                    tts.speak(tip, priority=2)
                
                asyncio.create_task(fetch_and_speak_tip())

            # Process voice agent queues
            await voice_agent.process_queries()

            # Prevent CPU pegging
            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
    finally:
        # Cleanup
        cap.release()
        face_detector.release()
        if object_detector:
            object_detector.stop()
        tts.stop()
        cv2.destroyAllWindows()
        
        # End session report
        await memory.end_session()
        print("[SYSTEM] Goodbye.")


if __name__ == "__main__":
    asyncio.run(async_main())
