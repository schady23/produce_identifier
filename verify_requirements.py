"""Reproducible assignment checks using actual inference; optional webcam and UI checks."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import unittest
import uuid
import numpy as np
import cv2
from pipeline_core import ContractError, INPUT_VERSION
from prototype import ROOT, DEFAULT_MODEL, YoloDetector, scan, append_log
from live_engine import LiveEngine, load_catalog


def run_verification(camera=None, ui=False, size=416):
    evidence=ROOT / "docs/evidence"
    evidence.mkdir(parents=True,exist_ok=True)
    report={"timestamp_unix_s":time.time(),"size":size,"checks":{},"failure_injections":[]}
    def record(name,details=True):
        report["checks"][name]={"passed":True,"details":details}
    test_stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result=unittest.TextTestRunner(stream=test_stream,verbosity=2).run(suite)
    (evidence / "tests.txt").write_text(test_stream.getvalue(),encoding="utf-8")
    if not result.wasSuccessful():
        raise AssertionError(test_stream.getvalue())
    record("contract_and_policy_tests",{"count":result.testsRun})
    detector=YoloDetector(str(DEFAULT_MODEL),"cpu")
    report["model"]=detector.manifest
    catalog=load_catalog()
    image=cv2.imread(str(ROOT / "samples/Granny-Smith-crop.jpg"))
    if image is None:
        raise RuntimeError("Missing public produce sample; see samples/SOURCES.md")
    event={"scan_id":str(uuid.uuid4()),"timestamp_unix_s":time.time(),"kind":"actual_model_baseline",
           "input_version":INPUT_VERSION}
    output=scan(image,"500","g",catalog,detector,size,.6,None,event)
    assert output["status"] == "confirmation_required", output
    assert output["candidates"][0]["label"] == "apple", output
    assert output["candidates"][0]["estimated_total_usd"] == "1.50"
    assert not output["confirmed"]
    for stage in ["validation_ms","transformation_ms","representation_ms","inference_ms","postprocessing_ms","policy_ms"]:
        assert stage in event
    event.update(status=output["status"],output=output)
    append_log(evidence / "actual_pipeline.jsonl",event)
    record("actual_model_pipeline",{"label":"apple","confidence":output["candidates"][0]["score"],"estimated_total_usd":"1.50"})
    elapsed=[]
    for _ in range(12):
        measured={"scan_id":str(uuid.uuid4()),"kind":"warmed_actual_inference","timestamp_unix_s":time.time()}
        warmed=scan(image,"500","g",catalog,detector,size,.6,None,measured)
        measured.update(status=warmed["status"],output=warmed)
        append_log(evidence / "actual_pipeline.jsonl",measured)
        elapsed.append(measured["service_latency_ms"])
    report["latency_ms"]={"count":len(elapsed),"p50":float(np.percentile(elapsed,50)),
                          "p90":float(np.percentile(elapsed,90)),"p95":float(np.percentile(elapsed,95)),
                          "p99":float(np.percentile(elapsed,99)),"scope":"warmed image-ready pipeline; excludes UI/model startup/camera/human wait"}
    record("stage_telemetry_and_latency",report["latency_ms"])
    for injection,boundary in [("bad-normalization","representation->inference"),
                               ("wrong-label-map","inference->postprocessing")]:
        failed={"scan_id":str(uuid.uuid4()),"timestamp_unix_s":time.time(),"injection":injection}
        try:
            scan(image,"500","g",catalog,detector,size,.6,injection,failed)
        except ContractError as exc:
            assert exc.boundary==boundary
            failed.update(status="error",boundary=exc.boundary,message=str(exc),error_type=type(exc).__name__)
            append_log(evidence / "actual_pipeline.jsonl",failed)
            report["failure_injections"].append({"injection":injection,"boundary":boundary,
                "outcome":"caught rejection; process continued; no silent acceptance"})
        else:
            raise AssertionError("Fault was not rejected: "+injection)
    recovered=scan(image,"500","g",catalog,detector,size,.6,None,{"scan_id":str(uuid.uuid4())})
    assert recovered["status"]=="confirmation_required"
    record("two_actual_failure_injections_and_recovery",report["failure_injections"])
    if camera is not None:
        cap=cv2.VideoCapture(camera,cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
        try:
            assert cap.isOpened(), "Camera could not open"
            camera_latencies=[]
            for index in range(5):
                ok,frame=cap.read()
                assert ok and frame is not None,"No camera frame"
                webcam_event={"scan_id":str(uuid.uuid4()),"timestamp_unix_s":time.time(),"kind":"webcam_verification","frame_number":index}
                webcam_output=scan(frame,None,"g",catalog,detector,size,.6,None,webcam_event,preview=True)
                assert not webcam_output["confirmed"]
                webcam_event.update(status=webcam_output["status"],output=webcam_output)
                append_log(ROOT / "logs/webcam_verification.jsonl",webcam_event)
                camera_latencies.append(webcam_event["service_latency_ms"])
            record("usb_camera_capture_and_actual_inference",{"camera_index":camera,"frames":5,"frame_shape":list(frame.shape),
                "mean_service_latency_ms":float(np.mean(camera_latencies)),"images_saved":False})
        finally:
            cap.release()
    if ui:
        os.environ["QT_QPA_PLATFORM"]="offscreen"
        from types import SimpleNamespace
        from PySide6.QtWidgets import QApplication
        from live_app import ProduceWindow
        app=QApplication.instance() or QApplication([])
        engine=LiveEngine(log=evidence / "ui_pipeline.jsonl",detector_factory=lambda *_: detector)
        args=SimpleNamespace(model=DEFAULT_MODEL,device="cpu",log=evidence / "ui_pipeline.jsonl",
            catalog=ROOT / "catalog.json",camera=0,image=None,no_camera=True)
        window=ProduceWindow(args,engine=engine)
        window.show()
        try:
            engine.start_image(image)
            deadline=time.monotonic()+10
            while engine.snapshot()["result"] is None and time.monotonic()<deadline:
                app.processEvents()
                time.sleep(.02)
            assert engine.snapshot()["result"] is not None
            window.tick()
            assert window.prediction.text()=="Apple"
            assert not window.confirm_button.isEnabled(),"Live prediction must not enable confirmation"
            window.weight.setText("500")
            deadline=time.monotonic()+5
            while engine.snapshot()["result"].settings_revision!=engine.snapshot()["revision"] and time.monotonic()<deadline:
                app.processEvents()
                time.sleep(.02)
            window.tick()
            window.toggle_pause()
            assert window.held is not None
            assert window.price.text()=="$1.50"
            assert window.confirm_button.isEnabled()
            window.confirm()
            assert window.confirmed
            assert not window.confirm_button.isEnabled()
            window.tick()
            app.processEvents()
            assert window.grab().save(str(evidence / "interface.png"))
            window.weight.setText("0")
            assert not window.confirm_button.isEnabled()
            assert window.price.text()=="--"
            window.toggle_pause()
            assert window.held is None
            record("qt_interface_actual_model_flow",{"recognition":True,"weight_entry":True,"freeze_before_confirm":True,
                "price":"1.50","confirmation_logged":True,"invalid_weight_disables_confirmation":True,
                "scope":"Qt component integration on offscreen display using a public sample; physical desktop visibility is separate"})
        finally:
            window.close()
            app.processEvents()
    report["passed"]=all(c["passed"] for c in report["checks"].values())
    (evidence / "verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera",type=int)
    parser.add_argument("--ui",action="store_true")
    parser.add_argument("--size",type=int,choices=[320,416,640],default=416)
    args=parser.parse_args()
    print(json.dumps(run_verification(args.camera,args.ui,args.size),indent=2))
