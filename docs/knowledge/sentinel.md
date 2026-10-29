---
title: Sentinel demonstration guide
source: Sentinel local product specification
revision: 2026-10-29
---
# Calibration
Calibrate this webcam demonstration while stationary, with even lighting and a frontal face. Keep eyes naturally open. Calibration requires at least 20 seconds of valid samples within 60 seconds. A calibration baseline adjusts the eye-closure threshold. Calibration does not establish detection accuracy.

# Limitations
This driver-monitoring demonstration measures visible facial geometry using a webcam. Lighting, glasses, occlusion, camera position, movement and tracking failures can affect measurements. Fatigue scores are fixed heuristic indicators, not medical diagnoses or validated predictions. Heart rate and three-minute forecasts are not provided.

# Synthetic scenarios
Synthetic replay data demonstrates alert and interface behavior. An artificial prolonged eye closure should produce a critical alert. A synthetic ten-minute report with 95 percent tracking coverage and two alerts describes artificial input only; it says nothing about a real driver. Reports must keep missing tracking visible and must never fabricate measurements.
