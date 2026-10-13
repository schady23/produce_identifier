# Produce Scanner

A working Windows desktop prototype with a live webcam view, real-time produce labels and bounding boxes, manually entered weight, an estimated price, and customer confirmation. Recognition runs locally; the camera and inference workers keep the interface responsive.

## Repository and video demo

- [GitHub: produce_identifier](https://github.com/schady23/produce_identifier)
- [YouTube demo: using the Python app with a banana, apple, and lemon](https://youtu.be/K5_JAtlxDCI)

The demo shows the prototype in use; the default model does not have a separate lemon class.

## AI assistance disclosure

I used Codex and GPT-6.1 to build the Python application and to assist with the prototype code.

## Start now

On a fresh checkout, follow **Setup on a fresh checkout** below to install the runtime and model. Then **double-click [start.cmd](start.cmd)** to open the interface with camera index 0.

Alternatively, run this command from the project folder:

```text
.\.venv\Scripts\python.exe live_app.py --camera 0
```

1. Place one loose produce item in view. Labels, scores, and boxes update continuously on the analyzed frame.
2. Enter its measured weight and select g, kg, or lb. Recognition also works with the weight field empty, but confirmation requires valid weight.
3. Click **Freeze frame**, review the image and estimated price, then **Confirm item**.
4. For low confidence or an ambiguous/missing model match, check **Select product manually** and select an active product. Confirming is a demo action using fictional SKUs and prices.
5. Click **Resume live** for the next item. Change the camera index and click Connect if a different camera opens. Stop releases the camera; closing the app also stops capture.

The default YOLO11 nano model recognizes **apple, banana, orange, carrot, and broccoli**. It does not recognize every item in the nine-item demo catalog. Other catalog items remain available for explicit manual selection. The broader Open Images nano model is also installed, but produced weaker category-specific results on our sample images. To compare it:

```text
.\.venv\Scripts\python.exe live_app.py --model models/yolov8n-oiv7.pt
```

Use a plain tray, steady lighting, and keep hands out of view where possible. Scores are uncalibrated; the 0.60 automatic-suggestion threshold is a prototype policy. A category does not establish variety or organic status. Detection can miss objects, so the one-item restriction also depends on the operator.

## Reproduce verification

```text
.\.venv\Scripts\python.exe verify_requirements.py --camera 0 --ui
```

Close the live interface before running camera verification so only one process owns the camera. This command runs contract/policy tests, actual-model inference on a public produce sample, both fault injections, warmed latency measurements, camera capture/inference, and Qt integration checks. UI integration uses an offscreen display and a public sample; it tests the interface behavior without saving your webcam images.

Latest verification: **30 tests passed**, actual apple prediction at **84% detector confidence**, correct **$1.50** estimate for 500 g at $3/kg, both injected faults caught, and normal inference recovered. Earlier [camera verification](docs/evidence/camera_verification.json) supplied five **640x480** frames from camera 0; mean camera-frame service time was **29.4 ms**. For 12 warmed public-image runs at size 416, pipeline p50/p95/p99 were **21.2/22.5/22.8 ms**. These are pipeline measurements, excluding camera/human waiting, UI rendering, startup, and frame-sampling intervals; they are not recognition accuracy benchmarks or edge-device timings.

[Requirement report](docs/part2_report.md) | [Machine-readable evidence](docs/evidence/verification.json) | [Interface preview](docs/evidence/interface.png) | [Test output](docs/evidence/tests.txt)

## Interface update

The interface uses a minimal paper/ink palette, a muted green accent, and code-drawn dither details. Dropdowns, settings, and the Qt file picker have explicit readable colors. Camera reconnects wait in a background worker, live frames cannot resize the control layout, and weight editing updates price without restarting recognition. Restart the app to load source changes.

## Pipeline failure controls

With the camera or sample preview running, click **Settings / pipeline checks** and choose **Inject normalization fault** or **Inject label-map fault**. The next frame is rejected with a named boundary and a JSONL log record; the app recovers on the next normal pass. Resume the preview before injecting a fault if the image is frozen.

The original single-scan CLI is still available:

```text
.\.venv\Scripts\python.exe prototype.py --image samples/Granny-Smith-crop.jpg --weight 500 --unit g
.\.venv\Scripts\python.exe prototype.py --image samples/Granny-Smith-crop.jpg --weight 500 --unit g --inject bad-normalization
.\.venv\Scripts\python.exe prototype.py --image samples/Granny-Smith-crop.jpg --weight 500 --unit g --inject wrong-label-map
```

## Setup on a fresh checkout

Use Windows and Python 3.10-3.12, then run `python install_runtime.py`. It creates `.venv`, installs the pinned CPU inference and Qt dependencies, and downloads/checks the official default checkpoint. Add `--include-broad-model` to download Open Images weights too. `--offline` uses the cached wheels available in this workspace, with already-installed dependencies as appropriate.

## Files and logs

- `live_app.py`: Qt interface, matched-frame overlays, live/frozen review, manual selection, and confirmation controls.
- `ui_theme.py`: application palette, popup styling, dither accents, and a video viewport with stable layout geometry.
- `live_engine.py`: single-slot frame buffer, camera/inference workers, versioned settings, error recovery, and confirmation checks.
- `prototype.py` / `pipeline_core.py`: shared pipeline, model adapter, contracts, unit conversion, catalog rules, and decimal pricing.
- `verify_requirements.py`: repeatable actual-model and hardware verification.
- `catalog.json`: nine demo items; all SKUs/prices are fictional. Multiple active SKUs for one label require manual selection.
- `docs/part2_report.md`: all seven assignment components, with measured evidence and remaining limitations.

Live statistics/errors and confirmations are appended to `logs/live.jsonl`. Runtime caches stay inside the project. The latest frame replaces earlier unprocessed frames; queues do not grow with camera FPS. Confirmations refer to a frozen analyzed frame, and current weight and active catalog are rechecked before saving. No webcam images are saved during normal use or verification. The exported interface preview uses the attributed public sample listed in [samples/SOURCES.md](samples/SOURCES.md).

The proposal/research documents preserve the earlier design decisions. The built prototype and the Part 2 report define the current implementation; custom training, a physical scale, full produce coverage, and edge deployment are future work.
