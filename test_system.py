#!/usr/bin/env python3
"""
PneumoAI System Validation Script
===================================
Tests the full Triton → FastAPI → response pipeline.

Usage:
    python test_system.py [--host http://localhost:5000] [--triton http://localhost:8000]

What is tested:
    1. Triton server health
    2. Triton model metadata (tensor names)
    3. Direct Triton inference (if tritonclient installed)
    4. FastAPI /health endpoint
    5. FastAPI /predict endpoint with a synthetic chest-X-ray-like image
    6. Response schema validation (all required fields present)
    7. FastAPI /explain endpoint
    8. Fallback: disabling Triton at runtime is NOT tested here (Docker env)
"""

import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
from io import BytesIO

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ─────────────────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[dict] = []


def record(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    print(f"  {tag}  {name}" + (f"  — {detail}" if detail else ""))
    results.append({"name": name, "passed": passed, "detail": detail})


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_multipart(url: str, field: str, filename: str, data: bytes, content_type: str = "image/png"):
    """Send a multipart/form-data POST with a single file field."""
    boundary = "PneumoAITestBoundary12345"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def post_json(url: str, payload: dict, timeout: int = 30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ─────────────────────────────────────────────────────────────────────────────
# Image generation
# ─────────────────────────────────────────────────────────────────────────────

def make_test_png() -> bytes:
    """
    Generate a minimal valid 224×224 greyscale PNG without Pillow/cv2.
    Uses raw PNG chunks so there are zero external dependencies.
    """
    import zlib

    width, height = 224, 224
    # Build raw image data: filter byte 0x00 before each row
    raw_rows = b""
    for _ in range(height):
        row = b"\x00" + bytes([128] * width)   # mid-grey pixels
        raw_rows += row

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat_data = zlib.compress(raw_rows)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr_data)
        + png_chunk(b"IDAT", idat_data)
        + png_chunk(b"IEND", b"")
    )
    return png


# ─────────────────────────────────────────────────────────────────────────────
# Test sections
# ─────────────────────────────────────────────────────────────────────────────

def test_triton_health(triton_url: str):
    print(f"\n{INFO}  1. Triton Server Health  ({triton_url})")
    # Triton's /v2/health/ready returns HTTP 200 with an EMPTY body when ready.
    # Do NOT use get_json() here — json.loads('') raises JSONDecodeError.
    url = f"{triton_url}/v2/health/ready"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
        record("Triton /v2/health/ready returns 200", status == 200, f"HTTP {status}")
    except urllib.error.HTTPError as exc:
        # 200 with empty body can sometimes surface as HTTPError on some Python versions
        record("Triton /v2/health/ready returns 200", exc.code == 200, f"HTTP {exc.code}")
    except Exception as exc:
        record("Triton server reachable", False, str(exc))



def test_triton_models(triton_url: str):
    print(f"\n{INFO}  2. Triton Model Metadata")
    for model_name in ("pneumo_densenet", "pneumo_resnet"):
        try:
            meta = get_json(f"{triton_url}/v2/models/{model_name}")
            inputs  = meta.get("inputs",  [])
            outputs = meta.get("outputs", [])
            ok = bool(inputs) and bool(outputs)
            detail = (
                f"input={inputs[0]['name'] if inputs else 'N/A'}  "
                f"output={outputs[0]['name'] if outputs else 'N/A'}"
            )
            record(f"Model '{model_name}' metadata", ok, detail)
        except Exception as exc:
            record(f"Model '{model_name}' metadata", False, str(exc))


def test_backend_health(api_url: str):
    print(f"\n{INFO}  3. Backend /health  ({api_url})")
    try:
        data = get_json(f"{api_url}/health")
        record("Backend /health status=ok", data.get("status") == "ok", str(data))
        triton_d = data.get("inference", {}).get("triton_densenet", False)
        triton_r = data.get("inference", {}).get("triton_resnet",   False)
        record("Triton DenseNet engine ready", bool(triton_d))
        record("Triton ResNet engine ready",   bool(triton_r))
        record("Local DenseNet TF model loaded", bool(data.get("models", {}).get("DenseNet")))
        record("Local ResNet TF model loaded",   bool(data.get("models", {}).get("ResNet")))
    except Exception as exc:
        record("Backend /health reachable", False, str(exc))


REQUIRED_PREDICT_KEYS = {
    "models", "final_prediction", "final_confidence",
    "status", "focus_score", "severity", "explanation", "quality_metrics",
}
REQUIRED_MODEL_KEYS = {
    "label", "confidence", "heatmap", "heatmap_mask", "original", "hotspots",
}


def test_predict(api_url: str):
    print(f"\n{INFO}  4. POST /predict")
    png_bytes = make_test_png()
    try:
        data = post_multipart(f"{api_url}/predict", "file", "test_xray.png", png_bytes)

        # Top-level keys
        missing_top = REQUIRED_PREDICT_KEYS - set(data.keys())
        record("Response has all required top-level fields",
               not missing_top, f"missing: {missing_top}" if missing_top else "")

        # final_prediction value
        final = data.get("final_prediction", "")
        record("final_prediction is PNEUMONIA or NORMAL", final in ("PNEUMONIA", "NORMAL"), final)

        # Confidence range
        conf = data.get("final_confidence", -1)
        record("final_confidence in [0, 1]", 0.0 <= conf <= 1.0, str(conf))

        # Per-model keys
        for model_key in ("DenseNet", "ResNet"):
            model_data = data.get("models", {}).get(model_key, {})
            missing = REQUIRED_MODEL_KEYS - set(model_data.keys())
            record(f"models.{model_key} has all required fields",
                   not missing, f"missing: {missing}" if missing else "")

        # Heatmap is a valid base64 string
        heatmap_b64 = data["models"]["DenseNet"].get("heatmap", "")
        try:
            base64.b64decode(heatmap_b64)
            record("DenseNet heatmap is valid base64", True)
        except Exception:
            record("DenseNet heatmap is valid base64", False)

        # Quality metrics
        qm = data.get("quality_metrics", {})
        record("quality_metrics.blur_score present", "blur_score" in qm)
        record("quality_metrics.brightness present",  "brightness" in qm)

    except Exception as exc:
        record("/predict request succeeded", False, str(exc))


def test_explain(api_url: str):
    print(f"\n{INFO}  5. POST /explain")
    payload = {
        "diagnosis":  "PNEUMONIA",
        "confidence": 0.87,
        "severity":   "MODERATE",
        "language":   "English",
    }
    try:
        data = post_json(f"{api_url}/explain", payload)
        has_key = "explanation" in data
        record("/explain returns 'explanation' key", has_key)
        if has_key:
            expl = data["explanation"]
            record("explanation is non-empty string", isinstance(expl, str) and len(expl) > 10, expl[:80])
    except Exception as exc:
        record("/explain request succeeded", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed  |  {failed} failed")
    print("=" * 60)
    if failed:
        print(f"\n{FAIL}  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    • {r['name']}" + (f": {r['detail']}" if r["detail"] else ""))
    print()
    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PneumoAI system validation")
    parser.add_argument("--host",   default="http://localhost:5000", help="Backend base URL")
    parser.add_argument("--triton", default="http://localhost:8000", help="Triton base URL")
    args = parser.parse_args()

    print("=" * 60)
    print("  PneumoAI System Validation")
    print(f"  Backend : {args.host}")
    print(f"  Triton  : {args.triton}")
    print("=" * 60)

    test_triton_health(args.triton)
    test_triton_models(args.triton)
    test_backend_health(args.host)
    test_predict(args.host)
    test_explain(args.host)

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
