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
