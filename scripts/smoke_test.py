"""Smoke test for a running API (local container or the cloud URL).

Usage:  python scripts/smoke_test.py http://localhost:8080
        python scripts/smoke_test.py https://YOUR-SERVICE-URL
Uses only the standard library.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def call(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
    malignant = json.loads((EXAMPLES / "sample_malignant.json").read_text())
    benign = json.loads((EXAMPLES / "sample_benign.json").read_text())
    checks = [
        ("GET /health", lambda: call(base, "/health"), lambda s, b: s == 200 and b.get("status") == "ok"),
        ("GET /model-info", lambda: call(base, "/model-info"), lambda s, b: s == 200 and len(b.get("features", [])) == 30),
        ("POST /predict (malignant sample)", lambda: call(base, "/predict", malignant), lambda s, b: s == 200 and b.get("prediction") == "malignant"),
        ("POST /predict (benign sample)", lambda: call(base, "/predict", benign), lambda s, b: s == 200 and b.get("prediction") == "benign"),
        ("POST /predict (bad input -> 400)", lambda: call(base, "/predict", {"features": [1, 2, 3]}), lambda s, b: s == 400 and "error" in b),
    ]
    failed = 0
    for name, run, ok in checks:
        try:
            status, body = run()
            passed = ok(status, body)
        except Exception as exc:  # connection errors etc.
            status, body, passed = "-", str(exc), False
        failed += not passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}  ->  {status} {json.dumps(body)[:110]}")
    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
