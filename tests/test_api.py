"""Tests for the prediction API. Run from the project root:  python -m unittest discover -s tests -v"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not (ROOT / "model" / "model.joblib").exists():          # train the model on first run
    subprocess.run([sys.executable, str(ROOT / "train.py")], check=True)

from app.main import app, FEATURES  # noqa: E402


def load(name):
    return json.loads((ROOT / "examples" / name).read_text())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"status": "ok"})

    def test_index_and_model_info(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        info = self.client.get("/model-info").get_json()
        self.assertEqual(len(info["features"]), 30)
        self.assertIn("test_metrics", info)

    def test_predict_malignant_with_named_features(self):
        r = self.client.post("/predict", json=load("sample_malignant.json"))
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["prediction"], "malignant")
        self.assertEqual(body["label"], 1)
        self.assertGreater(body["probability_malignant"], 0.9)

    def test_predict_benign_with_list(self):
        values = [load("sample_benign.json")["features"][f] for f in FEATURES]
        r = self.client.post("/predict", json={"features": values})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction"], "benign")
        self.assertLess(r.get_json()["probability_malignant"], 0.1)

    def test_batch(self):
        r = self.client.post("/predict/batch", json=load("sample_batch.json"))
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["count"], 2)
        self.assertEqual([p["prediction"] for p in body["predictions"]], ["malignant", "benign"])

    def test_missing_feature(self):
        feats = load("sample_malignant.json")["features"]
        feats.pop("mean radius")
        r = self.client.post("/predict", json={"features": feats})
        self.assertEqual(r.status_code, 400)
        self.assertIn("mean radius", r.get_json()["error"])

    def test_wrong_length_list(self):
        r = self.client.post("/predict", json={"features": [1, 2, 3]})
        self.assertEqual(r.status_code, 400)

    def test_non_numeric_value(self):
        feats = load("sample_malignant.json")["features"]
        feats["mean area"] = "large"
        self.assertEqual(self.client.post("/predict", json={"features": feats}).status_code, 400)

    def test_nan_rejected(self):
        raw = json.dumps(load("sample_malignant.json")).replace("17.99", "NaN")
        r = self.client.post("/predict", data=raw, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_not_json_and_missing_key(self):
        self.assertEqual(self.client.post("/predict", data="hello").status_code, 400)
        self.assertEqual(self.client.post("/predict", json={"x": 1}).status_code, 400)

    def test_empty_batch(self):
        self.assertEqual(self.client.post("/predict/batch", json={"instances": []}).status_code, 400)

    def test_unknown_route_and_wrong_method(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/predict").status_code, 405)


if __name__ == "__main__":
    unittest.main()
