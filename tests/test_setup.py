from core.setup import check_setup, verify_asset
from core.config import Settings


def test_model_checksum_and_optional_setup_without_camera(tmp_path):
    config = Settings(data_dir=tmp_path, model_path=tmp_path / "missing.task")
    result = check_setup(config)
    assert result["model"]["status"] == "missing"
    assert result["camera"]["status"] == "not probed"
    assert result["storage"]["status"] == "ready"
    assert not verify_asset(config.model_path)


def test_no_face_vision_worker_produces_no_biometrics():
    import time
    import numpy as np
    from perception.pipeline import FrameProcessor

    processor = FrameProcessor()
    try:
        result = processor.process((np.zeros((480, 640, 3), dtype=np.uint8), 1, time.time()))
        raw, preview = result
        assert raw["face_detected"] is False
        assert "ear" not in raw and "rppg_hr" not in raw
        assert preview
    finally:
        processor.cleanup()
        processor.worker.executor.shutdown()


def test_mouth_ratio_tracks_opening_not_lip_thickness():
    import numpy as np
    from perception.pipeline import _mar

    closed = np.array(
        [[0, 0], [100, 0], [50, -5], [50, 5], [0, 0], [0, 0], [50, -2], [50, 2]], dtype=float
    )
    opened = np.array(
        [[0, 0], [100, 0], [50, -35], [50, 35], [0, 0], [0, 0], [50, -32], [50, 32]], dtype=float
    )
    assert _mar(opened) > _mar(closed) * 5


def test_head_pose_frontal_geometry_is_zero_for_calibration():
    import cv2
    import numpy as np
    from perception.pipeline import _head_pose_euler

    face = np.array(
        [
            [0, 0, 0],
            [0, 330, 65],
            [-225, -170, 135],
            [225, -170, 135],
            [-150, 150, 125],
            [150, 150, 125],
        ],
        dtype=np.float64,
    )
    camera = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        face, np.zeros(3), np.array([0.0, 0.0, 1400.0]), camera, np.zeros(4)
    )
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for index, point in zip([1, 152, 226, 446, 57, 287], projected[:, 0]):
        landmarks[index, :2] = point / [640, 480]
    pitch, yaw, roll = _head_pose_euler(landmarks, (480, 640))
    assert abs(pitch) < 1 and abs(yaw) < 1 and abs(roll) < 1


def test_camera_frame_fit_preserves_aspect_ratio():
    import numpy as np
    from perception.pipeline import fit_frame

    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    fitted = fit_frame(frame, 640, 480)
    assert fitted.shape == (480, 640, 3)
    occupied = np.where(fitted[:, :, 0].max(axis=1) > 0)[0]
    assert len(occupied) == 360
