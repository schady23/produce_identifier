import json
import unittest
import tempfile
import time
import threading
from unittest.mock import patch
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pipeline_core import ContractError, decide, REP_VERSION
from prototype import letterbox_rgb, scan
from live_engine import LiveEngine, LatestFrame, Frame, Packet, confirm_snapshot, load_catalog

class LiveTests(unittest.TestCase):
    def packet(self, score=.9, extra=None):
        detections=[{"label":"apple","score":score,"box_model_px":[0,0,32,32]}]
        if extra:
            detections.append(extra)
        output={"scan_id":"held-frame","detections":detections,"confirmed":False,
                "geometry":{"scale_xy":[1,1],"pad_left_top":[0,0]},"candidates":[]}
        return Packet(Frame(np.zeros((32,32,3),dtype=np.uint8),1,0,1),output,{"threshold":.6},0)

    def test_actual_bgr_rgb_and_letterbox_geometry(self):
        frame=np.zeros((80,160,3),dtype=np.uint8)
        frame[:,:,2]=255
        image,geometry=letterbox_rgb(frame,320)
        self.assertEqual(tuple(image[160,160]),(255,0,0))
        self.assertEqual(tuple(image[0,0]),(114,114,114))
        self.assertEqual(geometry["pad_left_top"],[0,80])
        self.assertEqual(geometry["scale_xy"],[2,2])

    def test_latest_frame_queue_is_bounded_and_copies(self):
        slot=LatestFrame()
        image=np.zeros((32,32,3),dtype=np.uint8)
        for n in range(100):
            image.fill(n)
            slot.publish(image,3)
        image.fill(200)
        self.assertEqual(slot.latest().number,100)
        self.assertEqual(int(slot.latest().image[0,0,0]),99)
        self.assertEqual(slot.latest().session,3)

    def test_missing_weight_allows_preview_but_never_confirmation(self):
        result=decide(self.packet().output["detections"],None,load_catalog())
        self.assertEqual(result["status"],"weight_required")
        self.assertIsNone(result["weight_kg"])
        self.assertIsNone(result["candidates"][0]["estimated_total_usd"])
        self.assertFalse(result["confirmed"])
        with self.assertRaises(ContractError):
            confirm_snapshot(self.packet(),"","g",load_catalog(),"DEMO_APPLE")

    def test_confirmation_recalculates_changed_weight(self):
        result=confirm_snapshot(self.packet(),"750","g",load_catalog(),"DEMO_APPLE")
        self.assertEqual(result["selected"]["estimated_total_usd"],"2.25")
        self.assertEqual(result["weight_kg"],"0.750")
        self.assertTrue(result["confirmed"])

    def test_inactive_sku_is_rechecked_at_confirmation(self):
        catalog=load_catalog()
        catalog["items"][0]["active"]=False
        with self.assertRaises(ContractError):
            confirm_snapshot(self.packet(),"500","g",catalog,"DEMO_APPLE")

    def test_unrelated_sku_cannot_be_automatically_confirmed(self):
        with self.assertRaises(ContractError):
            confirm_snapshot(self.packet(),"500","g",load_catalog(),"DEMO_BANANA")

    def test_low_confidence_requires_explicit_manual_choice(self):
        with self.assertRaises(ContractError):
            confirm_snapshot(self.packet(.4),"500","g",load_catalog(),"DEMO_APPLE")
        result=confirm_snapshot(self.packet(.4),"500","g",load_catalog(),"DEMO_APPLE",manual=True)
        self.assertEqual(result["selection_mode"],"manual")

    def test_multiple_items_cannot_be_automatically_confirmed(self):
        packet=self.packet(extra={"label":"banana","score":.7})
        with self.assertRaises(ContractError):
            confirm_snapshot(packet,"500","g",load_catalog(),"DEMO_APPLE")

    def test_single_image_failure_injection_recovers_without_a_new_frame(self):
        class Detector:
            names={0:"Apple",1:"Banana"}
            from pipeline_core import label_hash
            map_hash=label_hash(names)
            manifest={"weights":"test fixture"}
            def infer(self,tensor,size):
                return np.array([[0,0,32,32,.9,0]],dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            engine=LiveEngine(log=Path(directory)/"live.jsonl",detector_factory=lambda *_:Detector())
            try:
                self.assertTrue(engine.ready.wait(2))
                engine.start_image(np.full((64,64,3),200,dtype=np.uint8))
                deadline=time.monotonic()+2
                while engine.snapshot()["result"] is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertIsNotNone(engine.snapshot()["result"])
                engine.inject("bad-normalization")
                deadline=time.monotonic()+2
                while (engine.snapshot()["errors"]<1 or engine.snapshot()["result"] is None) and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertEqual(engine.snapshot()["errors"],1)
                self.assertIsNotNone(engine.snapshot()["result"])
                self.assertEqual(engine.snapshot()["result"].output["status"],"weight_required")
                self.assertEqual(engine.snapshot()["error"]["boundary"],"representation->inference")
            finally:
                engine.close()

    def test_reconnect_click_waits_for_previous_camera_in_background(self):
        from pipeline_core import label_hash
        class Detector:
            names={0:"Apple",1:"Banana"}
            map_hash=label_hash(names)
            manifest={"weights":"fixture"}
            def infer(self,tensor,size):
                return np.array([[0,0,32,32,.9,0]],dtype=np.float32)
        first_read=threading.Event()
        unblock=threading.Event()
        next_read=threading.Event()
        counts={"owners":0,"max":0}
        lock=threading.Lock()
        class Camera:
            def __init__(self,index,*_):
                self.index=index
                with lock:
                    counts["owners"]+=1
                    counts["max"]=max(counts["max"],counts["owners"])
            def isOpened(self):
                return True
            def set(self,*_):
                return True
            def read(self):
                if self.index==0:
                    first_read.set()
                    unblock.wait(2)
                else:
                    next_read.set()
                    time.sleep(.01)
                return True,np.full((64,64,3),100,dtype=np.uint8)
            def release(self):
                with lock:
                    counts["owners"]-=1
        with tempfile.TemporaryDirectory() as folder, patch("cv2.VideoCapture",side_effect=Camera):
            engine=LiveEngine(log=Path(folder)/"live.jsonl",detector_factory=lambda *_:Detector())
            try:
                self.assertTrue(engine.ready.wait(2))
                engine.start_camera(0)
                self.assertTrue(first_read.wait(2))
                started=time.monotonic()
                engine.start_camera(1)
                self.assertLess(time.monotonic()-started,.3)
                self.assertFalse(next_read.is_set())
                unblock.set()
                self.assertTrue(next_read.wait(2))
                self.assertEqual(counts["max"],1)
                self.assertEqual(engine.snapshot()["source"],"webcam 1")
            finally:
                unblock.set()
                engine.close()
                engine.camera_thread.join(2)
                engine.worker.join(2)

if __name__=="__main__":
    unittest.main()
