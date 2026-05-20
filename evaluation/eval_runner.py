"""
DMS V5 Evaluation Framework
Measures detection accuracy, alert precision/recall, voice intent accuracy,
and memory retrieval quality against recorded ground truth sessions.

Run: uv run python -m evaluation.eval_runner
"""
from __future__ import annotations
import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from voice_agent.cognition import VoiceCognitionAgent, Intent

@dataclass
class DetectionGroundTruth:
    """Ground truth label for a single frame."""
    frame_id: int
    timestamp: float
    is_drowsy: bool
    is_yawning: bool
    is_distracted: bool
    fatigue_level: str
    ear_true: float
    notes: str = ""

@dataclass
class EvaluationResult:
    component: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    
    @property
    def precision(self) -> float:
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0
    
    @property
    def recall(self) -> float:
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def report(self) -> str:
        return (
            f"[{self.component}]\n"
            f"  Precision: {self.precision:.3f} | Recall: {self.recall:.3f} | F1: {self.f1:.3f}\n"
            f"  TP: {self.true_positives} FP: {self.false_positives} "
            f"FN: {self.false_negatives} TN: {self.true_negatives}\n"
            f"  Latency: mean={self.latency_ms_mean:.1f}ms p95={self.latency_ms_p95:.1f}ms"
        )

class DetectionEvaluator:
    """Evaluates detection accuracy against ground truth."""
    
    def __init__(self, ground_truth_path: str):
        self.gt_path = Path(ground_truth_path)
        self.ground_truth: List[DetectionGroundTruth] = []
    
    def load_ground_truth(self):
        if not self.gt_path.exists():
            print(f"[Eval] Ground truth file {self.gt_path} not found. Skipping load.")
            return
        with open(self.gt_path) as f:
            data = json.load(f)
        self.ground_truth = [DetectionGroundTruth(**d) for d in data["frames"]]
        print(f"[Eval] Loaded {len(self.ground_truth)} labeled frames")
    
    def evaluate_ear(self, predictions: List[Dict]) -> EvaluationResult:
        result = EvaluationResult("EAR_Drowsiness")
        latencies = []
        for gt, pred in zip(self.ground_truth, predictions):
            pred_drowsy = pred.get("ear_state") in ("warning", "critical")
            latencies.append(pred.get("ear_latency_ms", 0.0))
            if gt.is_drowsy and pred_drowsy:    result.true_positives += 1
            elif gt.is_drowsy and not pred_drowsy: result.false_negatives += 1
            elif not gt.is_drowsy and pred_drowsy: result.false_positives += 1
            else:                                  result.true_negatives += 1
        
        if latencies:
            result.latency_ms_mean = float(np.mean(latencies))
            result.latency_ms_p95 = float(np.percentile(latencies, 95))
        return result
    
    def evaluate_fatigue_score(self, predictions: List[Dict]) -> EvaluationResult:
        result = EvaluationResult("FatigueScore")
        for gt, pred in zip(self.ground_truth, predictions):
            pred_level = pred.get("fatigue_level", "normal")
            gt_positive = gt.fatigue_level in ("warning", "critical")
            pred_positive = pred_level in ("warning", "critical")
            
            if gt_positive and pred_positive:    result.true_positives += 1
            elif gt_positive and not pred_positive: result.false_negatives += 1
            elif not gt_positive and pred_positive: result.false_positives += 1
            else:                                   result.true_negatives += 1
        return result

class IntentEvaluator:
    """Evaluates voice agent intent classification accuracy."""
    
    INTENT_TEST_CASES = [
        ("how am I doing right now?",    "ask_current"),
        ("what was my fatigue like yesterday?", "ask_history"),
        ("am I getting worse?",          "ask_trend"),
        ("I'm feeling really sleepy",    "report_tired"),
        ("I'm fine, stop the alerts",    "report_fine"),
        ("you're too sensitive",         "adjust_threshold"),
        ("where can I pull over?",       "request_break"),
        ("what is the weather like?",    "unknown"),
    ]

    def evaluate(self, agent: VoiceCognitionAgent) -> EvaluationResult:
        result = EvaluationResult("IntentClassification")
        
        for text, expected_intent_val in self.INTENT_TEST_CASES:
            intent, conf = agent.classify_intent(text)
            if intent.value == expected_intent_val:
                result.true_positives += 1
            else:
                result.false_positives += 1
                print(f"[Intent Failure] '{text}' -> got {intent.value}, expected {expected_intent_val}")
                
        return result

async def run_evaluation():
    print("Starting DMS V5 offline evaluation...")
    
    # 1. Test Intent Routing
    agent = VoiceCognitionAgent(
        get_metrics=lambda: {}, get_memory_context=lambda: {}, 
        ollama_client=None, tts_engine=None
    )
    intent_eval = IntentEvaluator()
    res = intent_eval.evaluate(agent)
    print("\n--- Intent Evaluation ---")
    print(res.report())
    
    # 2. Test Synthetic Perception
    print("\n--- Perception Evaluation ---")
    from detectors.fatigue_score import FatigueScoreEngine
    fatigue_engine = FatigueScoreEngine()
    
    # Awake test — all signals normal
    total_fatigue = 0.0
    for i in range(30):
        ear_data = {"ear": 0.32, "drowsy_blink_score": 0.0}
        perclos_data = {"perclos": 0.05}
        mar_data = {"yawn_rate": 0}
        head_data = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "state": "normal", "sway_score": 0.1, "jerk": False, "distracted_frames": 0}
        gaze_data = {"gaze_x": 0.0, "state": "center", "counter": 0, "heatmap": np.zeros((60, 80)), "confidence": 1.0, "glasses_mode": False}
        rppg_data = {"hr_bpm": 70.0, "hr_valid": True, "hr_state": "normal"}
        
        fatigue_data = fatigue_engine.update(
            ear_data=ear_data, perclos_data=perclos_data, mar_data=mar_data,
            head_data=head_data, gaze_data=gaze_data, rppg_data=rppg_data
        )
        total_fatigue += fatigue_data["score"]
    
    awake_avg = total_fatigue / 30
    awake_pass = "PASS" if awake_avg < 30.0 else "FAIL"
    print(f"Awake synthetic: avg={awake_avg:.2f}/100 (expect <30) [{awake_pass}]")
    
    # Fatigued test — all signals firing
    fatigue_engine2 = FatigueScoreEngine()
    total_fatigue = 0.0
    for i in range(30):
        ear_data = {"ear": 0.15, "drowsy_blink_score": 0.8}
        perclos_data = {"perclos": 0.45}
        mar_data = {"yawn_rate": 5}
        head_data = {"pitch": 20.0, "yaw": 10.0, "roll": 0.0, "state": "nod_down", "sway_score": 15.0, "jerk": True, "distracted_frames": 30}
        gaze_data = {"gaze_x": 1.5, "state": "off_road", "counter": 30, "heatmap": np.zeros((60, 80)), "confidence": 1.0, "glasses_mode": False}
        rppg_data = {"hr_bpm": 120.0, "hr_valid": True, "hr_state": "stress"}
        
        fatigue_data = fatigue_engine2.update(
            ear_data=ear_data, perclos_data=perclos_data, mar_data=mar_data,
            head_data=head_data, gaze_data=gaze_data, rppg_data=rppg_data
        )
        total_fatigue += fatigue_data["score"]

    fatigued_avg = total_fatigue / 30
    fatigued_pass = "PASS" if fatigued_avg > 70.0 else "FAIL"
    print(f"Fatigued synthetic: avg={fatigued_avg:.2f}/100 (expect >70) [{fatigued_pass}]")
    print("\nEvaluation finished.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
