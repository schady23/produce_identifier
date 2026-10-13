"""Versioned contracts and checkout policy; independent of the model runtime."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import numpy as np

INPUT_VERSION = "scan-v1"
REP_VERSION = "rgb-nchw-01-v1"
OUTPUT_VERSION = "checkout-v1"
PRODUCE = {"apple", "artichoke", "banana", "bell pepper", "broccoli", "cabbage",
           "cantaloupe", "carrot", "coconut", "common fig", "cucumber", "garden asparagus",
           "grape", "grapefruit", "lemon", "mango", "mushroom", "orange", "peach", "pear",
           "pineapple", "pomegranate", "potato", "pumpkin", "radish", "squash (plant)",
           "strawberry", "tomato", "watermelon", "zucchini"}

class ContractError(ValueError):
    def __init__(self, boundary, message):
        self.boundary = boundary
        super().__init__(message)

def require(condition, boundary, message):
    if not condition:
        raise ContractError(boundary, message)

def weight_kg(value, unit):
    boundary = "input->validation"
    factors = {"kg": Decimal("1"), "g": Decimal("0.001"), "lb": Decimal("0.45359237")}
    require(unit in factors, boundary, "weight unit must be kg, g, or lb")
    try:
        result = Decimal(str(value)) * factors[unit]
    except (InvalidOperation, TypeError, ValueError):
        raise ContractError(boundary, "weight must be a finite number") from None
    require(result.is_finite() and Decimal("0") < result <= Decimal("20"),
            boundary, "weight must be in (0,20] kg")
    return result

def validate_frame(frame):
    boundary = "input->validation"
    require(isinstance(frame, np.ndarray), boundary, "missing or undecodable frame")
    require(frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3,
            boundary, "frame must be HWC BGR uint8 with 3 channels")
    h, w = frame.shape[:2]
    require(32 <= h <= 2160 and 32 <= w <= 3840, boundary, "unsupported frame dimensions")
    return frame

def validate_representation(tensor, version, size):
    boundary = "representation->inference"
    require(version == REP_VERSION, boundary, "preprocessing version mismatch")
    require(isinstance(tensor, np.ndarray) and tensor.dtype == np.float32,
            boundary, "tensor must be float32")
    require(tensor.shape == (1, 3, size, size), boundary, "tensor must be (1,3,S,S)")
    require(np.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1,
            boundary, "tensor must have finite normalized values in [0,1]")

def label_hash(names):
    return hashlib.sha256(json.dumps({str(k): v for k, v in names.items()},
                          sort_keys=True, ensure_ascii=True).encode()).hexdigest()

def validate_detections(rows, names, expected_hash, size):
    boundary = "inference->postprocessing"
    require(label_hash(names) == expected_hash, boundary, "label map version mismatch")
    require(isinstance(rows, np.ndarray) and rows.ndim == 2 and rows.shape[1] == 6,
            boundary, "expected Nx6 rows: x1,y1,x2,y2,score,class_id")
    require(len(rows) <= 300 and np.isfinite(rows).all(), boundary, "invalid detection count/values")
    detections = []
    for x1, y1, x2, y2, score, class_id in rows:
        require(0 <= score <= 1 and float(class_id).is_integer() and int(class_id) in names,
                boundary, "invalid confidence or class ID")
        require(0 <= x1 < x2 <= size and 0 <= y1 < y2 <= size,
                boundary, "invalid box coordinates (model pixels)")
        detections.append({"label": names[int(class_id)].lower(), "score": float(score),
                           "box_model_px": [float(x1), float(y1), float(x2), float(y2)]})
    return detections

def validate_catalog(catalog):
    boundary = "catalog->policy"
    require(isinstance(catalog, dict) and catalog.get("version") == "catalog-v1",
            boundary, "unsupported catalog version")
    require(catalog.get("currency") == "USD" and isinstance(catalog.get("items"), list)
            and 1 <= len(catalog["items"]) <= 100, boundary, "invalid catalog schema")
    ids = set()
    for item in catalog["items"]:
        require(isinstance(item, dict) and isinstance(item.get("sku"), str)
                and 1 <= len(item["sku"]) <= 40 and item["sku"] not in ids,
                boundary, "missing or duplicate SKU")
        ids.add(item["sku"])
        require(isinstance(item.get("label"), str) and item["label"] in PRODUCE
                and isinstance(item.get("name"), str) and 1 <= len(item["name"]) <= 80
                and type(item.get("active")) is bool, boundary, "invalid catalog item")
        try:
            price = Decimal(str(item.get("price_per_kg")))
        except InvalidOperation:
            raise ContractError(boundary, "invalid price") from None
        require(price.is_finite() and 0 < price <= 1000, boundary, "invalid USD/kg price")
    return catalog

def decide(detections, mass, catalog, threshold=0.60):
    validate_catalog(catalog)
    # Low-confidence produce still prevents pricing a weight containing multiple objects.
    visible = [d for d in detections if d["label"] in PRODUCE and d["score"] >= 0.25]
    base = {"version": OUTPUT_VERSION, "weight_kg": None if mass is None else str(mass), "currency": "USD",
            "catalog_version": catalog["version"], "confirmed": False,
            "candidates": [], "status": "manual_selection", "reason": "no_supported_produce"}
    if len(visible) != 1:
        if len(visible) > 1:
            base["reason"] = "multiple_produce_detections"
        return base
    detection = visible[0]
    items = [i for i in catalog["items"] if i["active"] and i["label"] == detection["label"]]
    if not items:
        base["reason"] = "not_in_active_catalog"
        return base
    for item in items:
        total = None if mass is None else (mass * Decimal(str(item["price_per_kg"]))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        base["candidates"].append({"sku": item["sku"], "name": item["name"],
                                   "label": item["label"], "score": detection["score"],
                                   "estimated_total_usd": None if total is None else str(total)})
    if detection["score"] < threshold:
        base["reason"] = "low_confidence"
    elif len(items) != 1:
        base["reason"] = "ambiguous_catalog_mapping"
    elif mass is None:
        base.update(status="weight_required", reason="enter_weight_to_estimate_price")
    else:
        base.update(status="confirmation_required", reason="one_active_match")
    return base
