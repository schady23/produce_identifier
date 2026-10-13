import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pipeline_core as core
import prototype

CATALOG = json.loads((Path(__file__).resolve().parents[1] / "catalog.json").read_text())

class FakeDetector:
    # Contract fixture, not evidence of YOLO accuracy or a real model call.
    names = {0: "Apple", 1: "Banana"}
    map_hash = core.label_hash(names)
    manifest = {"weights": "TEST FIXTURE"}
    def infer(self, tensor, size):
        return np.array([[0, 0, 20, 20, .9, 0]], dtype=np.float32)

class ContractTests(unittest.TestCase):
    def test_weight_units_are_equivalent(self):
        self.assertEqual(core.weight_kg(500, "g"), core.weight_kg("0.5", "kg"))
        self.assertEqual(core.weight_kg(1, "lb"), Decimal("0.45359237"))

    def test_invalid_weights_are_rejected(self):
        for weight, unit in [(None, "kg"), ("NaN", "kg"), ("inf", "kg"), (-1, "kg"), (0, "kg"), (21, "kg"), (2, "oz")]:
            with self.subTest(weight=weight, unit=unit), self.assertRaises(core.ContractError):
                core.weight_kg(weight, unit)

    def test_frame_contract_rejects_missing_grayscale_float(self):
        for frame in [None, np.zeros((64, 64), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.float32)]:
            with self.assertRaises(core.ContractError):
                core.validate_frame(frame)

    def test_tensor_contract_rejects_nan_shape_version(self):
        good = np.zeros((1, 3, 320, 320), dtype=np.float32)
        core.validate_representation(good, core.REP_VERSION, 320)
        bad = good.copy()
        bad[0, 0, 0, 0] = np.nan
        for tensor, version in [(bad, core.REP_VERSION), (good[0], core.REP_VERSION), (good, "old-v0")]:
            with self.assertRaises(core.ContractError):
                core.validate_representation(tensor, version, 320)

    def test_price_and_confirmation_requirement(self):
        result = core.decide([{"label": "apple", "score": .9}], Decimal(".5"), CATALOG)
        self.assertEqual(result["candidates"][0]["estimated_total_usd"], "1.50")
        self.assertEqual(result["status"], "confirmation_required")
        self.assertFalse(result["confirmed"])

    def test_empty_and_multiple_items_require_manual_selection(self):
        for detections in [[], [{"label": "apple", "score": .9}, {"label": "banana", "score": .3}]]:
            result = core.decide(detections, Decimal("1"), CATALOG)
            self.assertEqual(result["status"], "manual_selection")
            self.assertEqual(result["candidates"], [])

    def test_low_confidence_falls_back(self):
        result = core.decide([{"label": "apple", "score": .4}], Decimal("1"), CATALOG)
        self.assertEqual(result["reason"], "low_confidence")

    def test_inactive_or_ambiguous_sku_falls_back(self):
        catalog = json.loads(json.dumps(CATALOG))
        catalog["items"][0]["active"] = False
        result = core.decide([{"label": "apple", "score": .9}], Decimal("1"), catalog)
        self.assertEqual(result["reason"], "not_in_active_catalog")
        catalog["items"][0]["active"] = True
        extra = dict(catalog["items"][0], sku="DEMO_APPLE_ORGANIC")
        catalog["items"].append(extra)
        result = core.decide([{"label": "apple", "score": .9}], Decimal("1"), catalog)
        self.assertEqual(result["reason"], "ambiguous_catalog_mapping")

    def test_invalid_boxes_scores_class_ids_rejected(self):
        for row in [[0,0,20,20,1.1,0], [20,0,0,20,.9,0], [0,0,20,20,.9,2], [0,0,20,20,.9,.5]]:
            with self.assertRaises(core.ContractError):
                core.validate_detections(np.array([row]), FakeDetector.names, FakeDetector.map_hash, 320)

    def test_invalid_catalog_rejected(self):
        for price in ["NaN", "0", "-1", "not a number"]:
            catalog = json.loads(json.dumps(CATALOG))
            catalog["items"][0]["price_per_kg"] = price
            with self.assertRaises(core.ContractError):
                core.validate_catalog(catalog)

    def scan_fixture(self, injection, event):
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)
        canvas = np.full((320, 320, 3), 200, dtype=np.uint8)
        with patch.object(prototype, "letterbox_rgb", return_value=(canvas, {"fixture": True})):
            return prototype.scan(frame, "500", "g", CATALOG, FakeDetector(), 320, .6, injection, event)

    def test_valid_pipeline_fixture_and_json_logging(self):
        event = {"scan_id": "test"}
        result = self.scan_fixture(None, event)
        self.assertEqual(result["status"], "confirmation_required")
        self.assertIn("inference_ms", event)
        self.assertAlmostEqual(event["representation"]["max"], 200 / 255, places=6)
        with tempfile.TemporaryDirectory() as directory:
            logfile = Path(directory) / "scan.jsonl"
            prototype.append_log(logfile, event)
            self.assertEqual(json.loads(logfile.read_text())["scan_id"], "test")

    def test_failure_injection_bad_normalization(self):
        with self.assertRaises(core.ContractError) as caught:
            self.scan_fixture("bad-normalization", {"scan_id": "test"})
        self.assertEqual(caught.exception.boundary, "representation->inference")

    def test_failure_injection_wrong_label_map(self):
        with self.assertRaises(core.ContractError) as caught:
            self.scan_fixture("wrong-label-map", {"scan_id": "test"})
        self.assertEqual(caught.exception.boundary, "inference->postprocessing")

if __name__ == "__main__":
    unittest.main()
