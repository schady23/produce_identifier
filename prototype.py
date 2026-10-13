"""Single-scan Windows webcam prototype. Run python prototype.py --help."""
import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
import numpy as np
from pipeline_core import (ContractError, INPUT_VERSION, REP_VERSION, PRODUCE, require,
    validate_frame, validate_representation, weight_kg, label_hash,
    validate_detections, validate_catalog, decide)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/yolo11n.pt"

def letterbox_rgb(frame, size):
    import cv2
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = rgb
    return canvas, {"source_hw": [h, w], "scale_xy": [nw / w, nh / h],
                    "pad_left_top": [left, top], "model_size_px": size}

class YoloDetector:
    def __init__(self, weights, device):
        import os
        (ROOT / ".runtime/ultralytics").mkdir(parents=True, exist_ok=True)
        (ROOT / ".runtime/matplotlib").mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".runtime/ultralytics")
        os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime/matplotlib")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        from ultralytics import YOLO
        start = time.perf_counter()
        self.model = YOLO(weights)
        require(self.model.task == "detect", "model->startup", "use a detection checkpoint")
        self.names = dict(self.model.names)
        self.map_hash = label_hash(self.names)
        checkpoint = Path(getattr(self.model, "ckpt_path", None) or weights)
        self.manifest = {"weights": str(checkpoint), "label_map_sha256": self.map_hash,
                         "representation_version": REP_VERSION, "device": device,
                         "load_ms": (time.perf_counter() - start) * 1000}
        if checkpoint.is_file():
            self.manifest["weights_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        require(bool(PRODUCE.intersection(n.lower() for n in self.names.values())),
                "model->startup", "checkpoint has no recognized produce labels")
        self.device = device

    def infer(self, tensor, size):
        import torch
        # Tensor input is already RGB NCHW float32 [0,1]; no second normalization.
        results = self.model.predict(source=torch.from_numpy(tensor), imgsz=size,
            device=self.device, conf=0.25, iou=0.45, max_det=300, verbose=False)
        require(len(results) == 1 and results[0].boxes is not None,
                "inference->postprocessing", "missing detection result")
        require(label_hash(dict(results[0].names)) == self.map_hash,
                "inference->postprocessing", "model returned an unexpected label map")
        return results[0].boxes.data.cpu().numpy()

def acquire(args):
    import cv2
    if args.image:
        return cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    try:
        require(cap.isOpened(), "camera->input", "cannot open webcam; try --camera 1")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Place ONE loose produce item in view. SPACE: scan; Q: quit.")
        while True:
            ok, frame = cap.read()
            require(ok and frame is not None, "camera->input", "webcam frame unavailable")
            cv2.imshow("Produce scanner - SPACE to scan", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == 32:
                return frame
    finally:
        cap.release()
        cv2.destroyAllWindows()

def append_log(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")

def scan(frame, weight, unit, catalog, detector, size, threshold, injection, event, preview=False):
    start = time.perf_counter()
    stage_start = start
    require(size in (320, 416, 640), "config->representation", "unsupported inference size")
    require(np.isfinite(threshold) and 0.25 <= threshold <= 1, "config->policy", "invalid threshold")
    frame = validate_frame(frame)
    mass = None if preview and weight is None else weight_kg(weight, unit)
    event["mode"] = "preview" if preview else "checkout"
    validate_catalog(catalog)
    event["validation_ms"] = (time.perf_counter() - stage_start) * 1000
    event["input"] = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                      "bgr_mean": frame.mean(axis=(0, 1)).tolist(),
                      "bgr_std": frame.std(axis=(0, 1)).tolist(),
                      "weight_kg": None if mass is None else str(mass)}
    stage_start = time.perf_counter()
    canvas, geometry = letterbox_rgb(frame, size)
    event["transformation_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
    if injection == "bad-normalization":
        tensor = tensor * np.float32(255)  # Simulate forgetting the /255 normalization.
    event["representation"] = {"version": REP_VERSION, "shape": list(tensor.shape),
        "min": float(tensor.min()), "max": float(tensor.max()),
        "rgb_mean": tensor.mean(axis=(0, 2, 3)).tolist(),
        "rgb_std": tensor.std(axis=(0, 2, 3)).tolist(), "geometry": geometry}
    validate_representation(tensor, REP_VERSION, size)
    event["representation_ms"] = (time.perf_counter() - stage_start) * 1000
    event["transformation_representation_ms"] = event["transformation_ms"] + event["representation_ms"]
    stage_start = time.perf_counter()
    rows = detector.infer(tensor, size)
    event["inference_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()
    names = dict(detector.names)
    if injection == "wrong-label-map":
        keys = sorted(names)
        require(len(keys) >= 2, "injection->postprocessing", "need at least two labels")
        names[keys[0]], names[keys[1]] = names[keys[1]], names[keys[0]]
    detections = validate_detections(rows, names, detector.map_hash, size)
    event["detections"] = {"count": len(detections), "produce_count": sum(d["label"] in PRODUCE for d in detections),
                            "labels": [d["label"] for d in detections],
                            "scores": [d["score"] for d in detections]}
    event["postprocessing_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()
    output = decide(detections, mass, catalog, threshold)
    event["policy_ms"] = (time.perf_counter() - stage_start) * 1000
    event["postprocessing_policy_ms"] = event["postprocessing_ms"] + event["policy_ms"]
    event["service_latency_ms"] = (time.perf_counter() - start) * 1000
    output["detections"] = detections
    output["geometry"] = geometry
    output["scan_id"] = event["scan_id"]
    output["model"] = detector.manifest
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--camera", type=int, default=0)
    parser.add_argument("--weight", help="manually supplied mass; otherwise prompted after capture")
    parser.add_argument("--unit", choices=["kg", "g", "lb"], default="kg")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, choices=[320, 416, 640], default=640)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog.json")
    parser.add_argument("--log", type=Path, default=ROOT / "logs/scans.jsonl")
    parser.add_argument("--confirm", action="store_true", help="prompt to confirm an eligible demo selection")
    parser.add_argument("--inject", choices=["bad-normalization", "wrong-label-map"])
    args = parser.parse_args()
    event = {"scan_id": str(uuid.uuid4()), "timestamp_unix_s": time.time(),
             "input_version": INPUT_VERSION, "injection": args.inject}
    start = time.perf_counter()
    code = 0
    try:
        require(np.isfinite(args.threshold) and 0.25 <= args.threshold <= 1,
                "config->policy", "threshold must be in [0.25,1]")
        catalog = validate_catalog(json.loads(args.catalog.read_text(encoding="utf-8")))
        detector = YoloDetector(args.model, args.device)
        event["model"] = detector.manifest
        frame = acquire(args)
        weight = args.weight if args.weight is not None else input("Weight (" + args.unit + "): ").strip()
        result = scan(frame, weight, args.unit, catalog, detector, args.size, args.threshold, args.inject, event)
        print(json.dumps(result, indent=2))
        if args.confirm and result["status"] == "confirmation_required":
            if input("Confirm this demo selection? [y/N] ").strip().lower() == "y":
                result.update(status="confirmed_demo", confirmed=True)
                print(json.dumps(result, indent=2))
        event["output"] = result
    except KeyboardInterrupt:
        event.update(status="cancelled")
        code = 130
    except Exception as exc:
        event.update(status="error", boundary=getattr(exc, "boundary", "runtime"),
                     error_type=type(exc).__name__, message=str(exc))
        print(json.dumps({"status": "error", "boundary": event["boundary"], "message": str(exc)}), file=sys.stderr)
        code = 2
    event["session_elapsed_ms"] = (time.perf_counter() - start) * 1000
    try:
        append_log(args.log, event)
    except OSError as exc:
        print("Logging failed: " + str(exc), file=sys.stderr)
        code = 2
    return code

if __name__ == "__main__":
    raise SystemExit(main())
