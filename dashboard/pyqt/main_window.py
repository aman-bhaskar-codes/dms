import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
import time

class HUDPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.label = QLabel("HUD: Camera Feed & Overlays")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #000; color: #0f0; font-size: 20px;")
        layout.addWidget(self.label)

    def update_frame(self, frame):
        import cv2
        from PyQt6.QtGui import QImage, QPixmap
        if frame is not None:
            try:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = q_img.scaled(self.label.width(), self.label.height(), Qt.AspectRatioMode.KeepAspectRatio)
                self.label.setPixmap(QPixmap.fromImage(p))
            except Exception as e:
                pass

class AnalyticsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.score_label = QLabel("Fatigue Score: 0")
        self.score_label.setStyleSheet("color: white; font-size: 16px;")
        self.level_label = QLabel("Level: NORMAL")
        self.level_label.setStyleSheet("color: #10B981; font-size: 18px; font-weight: bold;")
        
        layout.addWidget(self.score_label)
        layout.addWidget(self.level_label)
        layout.addStretch()

    def update_metrics(self, state):
        score = state.get("fatigue_score", 0.0)
        lvl = state.get("fatigue_level", "normal")
        self.score_label.setText(f"Fatigue Score: {score:.1f}")
        self.level_label.setText(f"Level: {lvl.upper()}")
        
        colors = {
            "normal": "#10B981",
            "mild": "#F59E0B",
            "warning": "#EF4444",
            "critical": "#DC2626"
        }
        self.level_label.setStyleSheet(f"color: {colors.get(lvl, '#ffffff')}; font-size: 18px; font-weight: bold;")

class TopBarWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self.title = QLabel("DMS V3 — Driver Monitoring System")
        self.title.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.title)

class DMSMainWindow(QMainWindow):
    def __init__(self, state_dict=None):
        super().__init__()
        self.state_dict = state_dict or {}
        self.setWindowTitle("DMS V3")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: #0A0E1A;")

        central = QWidget()
        main_layout = QVBoxLayout(central)

        main_layout.addWidget(TopBarWidget(), stretch=0)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.hud = HUDPanel()
        self.analytics = AnalyticsPanel()
        
        top_splitter.addWidget(self.hud)
        top_splitter.addWidget(self.analytics)
        top_splitter.setSizes([800, 400])
        main_layout.addWidget(top_splitter, stretch=1)

        self.setCentralWidget(central)
        
        # Polling UI update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(500) # 2 Hz refresh

    def refresh_ui(self):
        self.analytics.update_metrics(self.state_dict)
        if "frame" in self.state_dict:
            self.hud.update_frame(self.state_dict["frame"])

def run_gui(state_dict):
    app = QApplication(sys.argv)
    window = DMSMainWindow(state_dict)
    window.show()
    sys.exit(app.exec())
