"""Bounded live pipeline, independent of the desktop interface."""
from dataclasses import dataclass, replace
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from pipeline_core import (ContractError, INPUT_VERSION, PRODUCE, require,
                           validate_frame, validate_catalog, weight_kg, decide)
from prototype import ROOT, DEFAULT_MODEL, YoloDetector, scan, append_log

@dataclass(frozen=True)
class Settings:
    weight: str = ""
    unit: str = "g"
    size: int = 416
    threshold: float = 0.60

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    number: int
    captured_at: float
    session: int

@dataclass(frozen=True)
class Packet:
    frame: Frame
    output: dict
    event: dict
    settings_revision: int

class LatestFrame:
    """One slot; a slow consumer cannot create an unbounded frame queue."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.number = 0

    def publish(self, image, session):
        with self.lock:
            self.number += 1
            self.frame = Frame(image.copy(), self.number, time.monotonic(), session)
            return self.frame

    def latest(self):
        with self.lock:
            return self.frame

    def clear(self):
        with self.lock:
            self.frame = None


def load_catalog(path=ROOT / "catalog.json"):
    return validate_catalog(json.loads(Path(path).read_text(encoding="utf-8")))


def confirm_snapshot(packet, weight, unit, catalog, sku, manual=False):
    """Recheck current mass/catalog against the held image before confirmation."""
    mass = weight_kg(weight, unit)
    validate_catalog(catalog)
    policy = decide(packet.output["detections"], mass, catalog,
                    packet.event.get("threshold", 0.60))
    if not manual:
        require(policy["status"] == "confirmation_required", "policy->confirmation",
                "prediction requires manual product selection")
        allowed = [c["sku"] for c in policy["candidates"]]
        require(sku in allowed, "policy->confirmation", "SKU does not match the prediction")
    items = [i for i in catalog["items"] if i["active"] and i["sku"] == sku]
    require(len(items) == 1, "catalog->confirmation", "select one active catalog item")
    # Explicit manual choice supplies identity; mass still comes from validated input.
    priced = decide([{"label": items[0]["label"], "score": 1.0}], mass,
                    dict(catalog, items=items), 0.60)["candidates"][0]
    result = deepcopy(packet.output)
    result.update(status="confirmed_demo", confirmed=True, weight_kg=str(mass),
                  selected=priced, selection_mode="manual" if manual else "prediction",
                  catalog_version=catalog["version"])
    return result


class LiveEngine:
    def __init__(self, model=DEFAULT_MODEL, device="cpu", log=ROOT / "logs/live.jsonl",
                 catalog_path=ROOT / "catalog.json", detector_factory=YoloDetector):
        self.model_path, self.device = model, device
        self.log_path, self.catalog_path = Path(log), Path(catalog_path)
        self.detector_factory = detector_factory
        self.frames = LatestFrame()
        self.lock, self.log_lock = threading.Lock(), threading.Lock()
        self.closed = threading.Event()
        self.camera_stop = threading.Event()
        self.camera_owner = threading.Lock()
        self.ready = threading.Event()
        self.settings, self.revision = Settings(), 0
        self.session, self.active, self.paused = 0, False, False
        self.result, self.error, self.detector = None, None, None
        self.status = "Loading produce model..."
        self.source = "none"
        self.pending_injection = None
        self.errors, self.completed, self.frames_skipped = 0, 0, 0
        self.latencies = deque(maxlen=200)
        self.camera_thread = None
        self.worker = threading.Thread(target=self._infer_loop, daemon=True, name="produce-inference")
        self.worker.start()

    def set_settings(self, settings):
        with self.lock:
            if settings != self.settings:
                self.settings = settings
                self.revision += 1

    def snapshot(self):
        with self.lock:
            return {"result": self.result, "error": deepcopy(self.error), "status": self.status,
                    "ready": self.ready.is_set(), "active": self.active, "paused": self.paused,
                    "source": self.source, "session": self.session, "revision": self.revision,
                    "completed": self.completed, "errors": self.errors,
                    "frames_skipped": self.frames_skipped, "latencies": list(self.latencies)}

    def _log(self, event):
        try:
            with self.log_lock:
                append_log(self.log_path, event)
            return True
        except (OSError, ValueError) as exc:
            with self.lock:
                self.error = {"boundary": "output->logging", "message": str(exc),
                              "scan_id": event.get("scan_id"), "injection": event.get("injection")}
                self.status = "Cannot save pipeline log"
            return False

    def _error(self, exc, event):
        event.update(status="error", boundary=getattr(exc, "boundary", "runtime"),
                     error_type=type(exc).__name__, message=str(exc))
        with self.lock:
            self.errors += 1
            self.error = deepcopy(event)
        self._log(event)

    def stop_camera(self):
        self.camera_stop.set()
        with self.lock:
            self.active, self.paused = False, False
            self.session += 1
            self.result = None
            self.source = "none"
            self.pending_injection = None
            if self.ready.is_set():
                self.status = "Camera stopped"
        self.frames.clear()

    def start_image(self, image):
        validate_frame(image)
        self.stop_camera()
        with self.lock:
            self.active, self.paused, self.source = True, False, "sample image"
            self.status = "Analyzing image..."
            session = self.session
        self.frames.publish(image, session)

    def start_camera(self, index=0):
        # Serialize camera ownership in the capture worker, never block/reject a UI click.
        self.stop_camera()
        self.camera_stop = threading.Event()
        with self.lock:
            self.active, self.paused = True, False
            self.source = "webcam " + str(index)
            self.status = "Connecting camera..."
            session = self.session
        self.camera_thread = threading.Thread(target=self._capture_loop,
            args=(index, session, self.camera_stop), daemon=True, name="produce-camera")
        self.camera_thread.start()

    def hold_result(self):
        """Atomically hold a valid image; pricing is rechecked independently."""
        with self.lock:
            if not self.active or self.result is None or self.result.frame.session != self.session:
                return None
            self.paused = True
            return self.result

    def set_paused(self, paused):
        with self.lock:
            self.paused = paused

    def inject(self, name):
        require(name in ("bad-normalization", "wrong-label-map"), "test->pipeline", "unknown injection")
        with self.lock:
            require(self.active and not self.paused, "test->pipeline", "start or resume the preview first")
            self.pending_injection = name

    def _capture_loop(self, index, session, stop):
        import cv2
        cap = None
        owns_camera = False
        try:
            deadline = time.monotonic() + 5
            while not stop.is_set() and not self.closed.is_set():
                if self.camera_owner.acquire(timeout=0.05):
                    owns_camera = True
                    break
                require(time.monotonic() < deadline, "camera->input", "Camera is taking too long to close; retry Connect")
            if not owns_camera or stop.is_set() or self.closed.is_set():
                return
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF) if __import__("sys").platform == "win32" else (cv2.CAP_ANY,):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    break
                cap.release()
            require(cap is not None and cap.isOpened(), "camera->input",
                    "Cannot open camera. Close other camera apps or try another camera index.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while not stop.is_set() and not self.closed.is_set():
                ok, image = cap.read()
                require(ok and image is not None, "camera->input", "Camera disconnected or returned no frame")
                validate_frame(image)
                if not stop.is_set():
                    self.frames.publish(image, session)
        except Exception as exc:
            if not stop.is_set():
                self._error(exc, {"scan_id": str(uuid.uuid4()), "timestamp_unix_s": time.time(),
                                  "source": "webcam", "session": session})
                with self.lock:
                    if self.session == session:
                        self.active = False
                        self.result = None
                        self.status = str(exc)
        finally:
            if cap is not None:
                cap.release()
            if owns_camera:
                self.camera_owner.release()

    def _infer_loop(self):
        try:
            self.detector = self.detector_factory(str(self.model_path), self.device)
            with self.lock:
                self.status = "Model ready"
            self.ready.set()
        except Exception as exc:
            self._error(exc, {"scan_id": str(uuid.uuid4()), "timestamp_unix_s": time.time(),
                              "boundary": "model->startup"})
            with self.lock:
                self.status = "Model could not load: " + str(exc)
            return
        last_key, last_number = None, 0
        while not self.closed.is_set():
            with self.lock:
                settings, revision = self.settings, self.revision
                active, paused, session = self.active, self.paused, self.session
                injection = self.pending_injection
            frame = self.frames.latest()
            key = (frame.number, revision) if frame else None
            if not active or paused or frame is None or frame.session != session or (key == last_key and injection is None):
                self.closed.wait(0.03)
                continue
            with self.lock:
                if self.pending_injection == injection:
                    self.pending_injection = None
            last_key = key
            skipped = max(0, frame.number - last_number - 1) if last_number else 0
            last_number = frame.number
            start = time.perf_counter()
            event = {"scan_id": str(uuid.uuid4()), "timestamp_unix_s": time.time(),
                     "input_version": INPUT_VERSION, "injection": injection, "frame_number": frame.number,
                     "session": session, "settings_revision": revision, "source": self.source,
                     "frame_age_ms": (time.monotonic() - frame.captured_at) * 1000,
                     "frames_skipped": skipped, "threshold": settings.threshold,
                     "model": self.detector.manifest}
            try:
                catalog = load_catalog(self.catalog_path)
                output = scan(frame.image, settings.weight.strip() or None, settings.unit, catalog,
                              self.detector, settings.size, settings.threshold, injection, event, preview=True)
                event.update(status=output["status"], output=output)
                if self._log(event):
                    with self.lock:
                        self.completed += 1
                        self.frames_skipped += skipped
                        self.latencies.append(event["service_latency_ms"])
                        if session == self.session and revision == self.revision and not self.paused and self.active:
                            self.result = Packet(frame, output, event, revision)
                            self.status = "Live recognition" if self.source.startswith("webcam") else "Sample image analyzed"
            except Exception as exc:
                if injection:
                    last_key = None  # Re-run the unchanged sample normally after a one-shot fault.
                event["service_latency_ms"] = (time.perf_counter() - start) * 1000
                self._error(exc, event)
                with self.lock:
                    if session == self.session and revision == self.revision and not self.paused:
                        self.result = None
                        self.status = "Pipeline rejected this frame; correct the input or retry"
            self.closed.wait(0.08)

    def close(self):
        self.closed.set()
        self.stop_camera()
