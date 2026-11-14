"""Flask API that serves the trained breast-cancer classifier.

Endpoints
  GET  /              service description
  GET  /health        liveness check
  GET  /model-info    model description, feature names and test metrics
  POST /predict       one prediction   body: {"features": {name: value, ...}}  or {"features": [30 numbers]}
  POST /predict/batch many predictions body: {"instances": [ {...} or [...], ... ]}
"""
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
MAX_BATCH = 1000
LABELS = {0: "benign", 1: "malignant"}

model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = meta["features"]

app = Flask(__name__)


class InputError(ValueError):
    """Raised when the request body is not valid; becomes an HTTP 400 response."""


def parse_instance(obj, where="features"):
    """Turn a dict (by feature name) or a list (in feature order) into a list of 30 floats."""
    if isinstance(obj, dict):
        missing = [f for f in FEATURES if f not in obj]
        unknown = [k for k in obj if k not in FEATURES]
        if missing or unknown:
            raise InputError(f"{where}: missing features {missing}, unknown features {unknown}")
        values = [obj[f] for f in FEATURES]
    elif isinstance(obj, list):
        if len(obj) != len(FEATURES):
            raise InputError(f"{where}: expected {len(FEATURES)} values, got {len(obj)}")
        values = obj
    else:
        raise InputError(f"{where} must be an object of named features or a list of {len(FEATURES)} numbers")
    clean = []
    for name, v in zip(FEATURES, values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise InputError(f"{where}: value for '{name}' must be a finite number, got {v!r}")
        clean.append(float(v))
    return clean


def predict_rows(rows):
    probs = model.predict_proba(np.array(rows))[:, 1]
    return [{"prediction": LABELS[int(p >= 0.5)], "label": int(p >= 0.5),
             "probability_malignant": round(float(p), 4)} for p in probs]


def json_body():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise InputError("Request body must be a JSON object (send Content-Type: application/json)")
    return body


@app.get("/")
def index():
    return jsonify({"service": "Breast cancer classification API",
                    "endpoints": {"GET /health": "liveness check", "GET /model-info": "model details",
                                  "POST /predict": "single prediction", "POST /predict/batch": "batch prediction"},
                    "note": "Educational demo. Not a medical device."})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model-info")
def model_info():
    return jsonify(meta)


@app.post("/predict")
def predict():
    body = json_body()
    if "features" not in body:
        raise InputError("Body must contain a 'features' key")
    return jsonify(predict_rows([parse_instance(body["features"])])[0])


@app.post("/predict/batch")
def predict_batch():
    body = json_body()
    instances = body.get("instances")
    if not isinstance(instances, list) or not instances:
        raise InputError("Body must contain a non-empty 'instances' list")
    if len(instances) > MAX_BATCH:
        raise InputError(f"At most {MAX_BATCH} instances per request")
    rows = [parse_instance(inst, where=f"instances[{i}]") for i, inst in enumerate(instances)]
    return jsonify({"count": len(rows), "predictions": predict_rows(rows)})


@app.errorhandler(InputError)
def handle_input_error(err):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(404)
def handle_404(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def handle_405(_):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def handle_500(_):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":      # local development only; the container uses gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
