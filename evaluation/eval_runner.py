"""
Evaluation Framework for DMS V5
Runs offline datasets through the pipeline for regression testing.
"""
import asyncio
import cv2
import numpy as np
from detectors.fatigue_score import FatigueScoreEngine
from detectors.head_pose import HeadPoseDetector
from detectors.gaze import GazeTracker
from detectors.rppg import RPPGEstimator
from detectors.face_detector import FaceDetector

class MockDatabase:
    async def log_event(self, *args): pass
    async def log_ai_tip(self, *args): pass

class MockMemory:
    def __init__(self):
        self.session_id = 1
        self.total_fatigue = 0.0
    def update_fatigue(self, score):
        self.total_fatigue += score
    async def log_event(self, *args): pass
    async def get_session_context(self): return "Mock Context"

async def run_evaluation():
    print("Starting DMS V5 offline evaluation...")
    
    # Initialize components
    face_det = FaceDetector()
    head_pose = HeadPoseDetector()
    gaze = GazeTracker()
    rppg = RPPGEstimator()
    fatigue_engine = FatigueScoreEngine()
    
    memory = MockMemory()
    
    # Generate synthetic mock frames (simulating an awake driver)
    print("Running awake test...")
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mock face
        face_det.process(frame) # would normally process real frame
        
        # Override detections for mock
        head_data = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "state": "normal", "sway_score": 0.1, "jerk": False, "distracted_frames": 0}
        gaze_data = {"gaze_x": 0.0, "state": "center", "counter": 0, "heatmap": np.zeros((60, 80)), "confidence": 1.0, "glasses_mode": False}
        rppg_data = {"hr_bpm": 70.0, "hr_valid": True, "hr_state": "normal"}
        
        fatigue_data = fatigue_engine.update(
            ear_data={}, perclos_data={}, mar_data={},
            head_data=head_data, gaze_data=gaze_data, rppg_data=rppg_data
        )
        memory.update_fatigue(fatigue_data["score"])
        
    print(f"Awake test completed. Avg fatigue: {memory.total_fatigue / 30:.2f} (Expected < 30.0)")

    # Run fatigued test (distracted + swaying)
    print("Running fatigued test...")
    memory.total_fatigue = 0.0
    for i in range(30):
        head_data = {"pitch": 20.0, "yaw": 10.0, "roll": 0.0, "state": "nod_down", "sway_score": 15.0, "jerk": True, "distracted_frames": 30}
        gaze_data = {"gaze_x": 1.5, "state": "off_road", "counter": 30, "heatmap": np.zeros((60, 80)), "confidence": 1.0, "glasses_mode": False}
        rppg_data = {"hr_bpm": 120.0, "hr_valid": True, "hr_state": "stress"}
        
        fatigue_data = fatigue_engine.update(
            ear_data={}, perclos_data={}, mar_data={},
            head_data=head_data, gaze_data=gaze_data, rppg_data=rppg_data
        )
        memory.update_fatigue(fatigue_data["score"])

    print(f"Fatigued test completed. Avg fatigue: {memory.total_fatigue / 30:.2f} (Expected > 70.0)")
    print("Evaluation finished.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
