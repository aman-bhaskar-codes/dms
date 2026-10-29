"""Explicit checks and model installation. Importing this module never downloads anything."""

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from core.config import ROOT, Settings


def manifest():
    return json.loads((ROOT / "models/manifest.json").read_text())


def verify_asset(path):
    path = Path(path)
    entry = manifest()["face_landmarker"]
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    )


def check_setup(config=None, probe=False):
    config = config or Settings()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=config.data_dir) as f:
            f.write(b"sentinel storage check")
        storage = "ready"
    except OSError:
        storage = "not writable"
    result = dict(
        python={"status": "ready" if sys.version_info[:2] == (3, 12) else "requires Python 3.12"},
        model={
            "status": "ready"
            if verify_asset(config.model_path)
            else "checksum mismatch"
            if config.model_path.is_file()
            else "missing",
            "action": "sentinel setup-models",
        },
        storage={"status": storage},
        camera={
            "status": "not probed",
            "action": "Start live monitoring to request camera permission in macOS Privacy & Security",
        },
        microphone={
            "status": "not probed",
            "action": "Use Record locally to request microphone permission",
        },
        audio={"status": "ready" if shutil.which("say") else "macOS speech unavailable"},
        gemini={
            "status": "configured, not contacted"
            if config.gemini_api_key.get_secret_value()
            else "not configured"
        },
        knowledge={"status": "installed" if config.embedding_path.is_dir() else "lexical fallback"},
        voice={
            "status": "installed"
            if config.whisper_path.is_dir() and importlib.util.find_spec("faster_whisper")
            else "not installed"
        },
    )
    if probe:
        import cv2

        cap = cv2.VideoCapture(config.camera_index)
        try:
            ok, _ = cap.read()
            result["camera"]["status"] = (
                "ready" if ok else "permission denied or device unavailable"
            )
        finally:
            cap.release()
    return result


def setup_models(config, knowledge=False, voice=False):
    import urllib.request

    entry = manifest()["face_landmarker"]
    if not verify_asset(config.model_path):
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = config.model_path.with_suffix(".download")
        try:
            urllib.request.urlretrieve(entry["source"], temporary)
            if not verify_asset(temporary):
                raise ValueError("Downloaded face model checksum mismatch")
            temporary.replace(config.model_path)
        finally:
            temporary.unlink(missing_ok=True)
    installed = {}
    for name, enabled, target in [
        ("embeddings", knowledge, config.embedding_path),
        ("voice", voice, config.whisper_path),
    ]:
        if not enabled:
            continue
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(manifest()[name]["repository"])
        snapshot_download(
            repo_id=manifest()[name]["repository"],
            revision=info.sha,
            local_dir=target,
            allow_patterns=(
                ["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "README.md", "LICENSE*"]
                if name == "embeddings"
                else ["model.bin", "*.json", "*.txt", "README.md", "LICENSE*"]
            ),
        )
        installed[name] = dict(
            repository=manifest()[name]["repository"],
            revision=info.sha,
            license=manifest()[name]["license"],
            files={
                str(p.relative_to(target)): {
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "size": p.stat().st_size,
                }
                for p in target.rglob("*")
                if p.is_file() and ".cache" not in p.parts
            },
        )
    if installed:
        target = ROOT / "models/installed-models.json"
        previous = json.loads(target.read_text()) if target.exists() else {}
        target.write_text(json.dumps(previous | installed, indent=2) + "\n")
    return check_setup(config)
