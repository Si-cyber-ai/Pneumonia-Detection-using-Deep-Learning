# PneumoAI: Pneumonia Detection Using Deep Learning

PneumoAI is a full-stack chest X-ray screening workspace built around a dual-model deep learning ensemble. It combines real image inference, Grad-CAM explainability, image quality checks, AI-generated narrative support, and exportable clinical reporting — all powered by NVIDIA Triton Inference Server.

> This repository is intended for educational and research use only. It is not a medical device and must not be used as a substitute for licensed clinical judgment.

---

## SDG Alignment — SDG 3: Good Health and Well-Being

This project directly supports **United Nations Sustainable Development Goal 3 (SDG 3)**: *Ensure healthy lives and promote well-being for all at all ages*.

| SDG 3 Target | How PneumoAI Contributes |
|---|---|
| **3.8** Universal health coverage | AI-assisted screening reduces radiologist workload in resource-constrained settings |
| **3.d** Strengthen health early-warning capacity | Automated pneumonia detection enables faster clinical triage |
| **3.b** Access to medicines and vaccines | Rapid diagnosis supports early treatment initiation |

By deploying an open-source, Dockerised, production-grade ML system, PneumoAI reduces the barrier to accurate chest X-ray interpretation in clinics without specialist radiologists.

---

## Highlights

- Dual-model inference using DenseNet121 and ResNet50
- NVIDIA Triton Inference Server as primary inference engine (production-grade, batching-capable)
- Automatic fallback to local TensorFlow if Triton is unreachable — zero crashes
- FastAPI backend with Grad-CAM++ explainability
- React + Vite clinical review UI (unchanged)
- Groq-powered clinical explanation endpoint with graceful fallback
- Export-ready PDF clinical report generation
- Full Docker Compose multi-service deployment

---

## Screenshots

### Study intake and upload flow

![Study intake screen](./screenshots/Screenshot%202026-05-02%20172941.png)

### Prediction summary and quality metrics

![Prediction summary](./screenshots/Screenshot%202026-05-02%20172949.png)

### Grad-CAM comparison view

![Grad-CAM comparison](./screenshots/Screenshot%202026-05-02%20173007.png)

### Grad-CAM split slider view

![Grad-CAM split slider](./screenshots/Screenshot%202026-05-02%20173020.png)

### Exported clinical report preview

![Clinical report preview](./screenshots/Screenshot%202026-05-02%20173051.png)

---

## Inference Pipeline Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│                        PneumoAI Architecture                       │
└────────────────────────────────────────────────────────────────────┘

Browser (React + Vite)
        │
        │  HTTP (multipart/form-data)
        ▼
  Nginx Reverse Proxy  (:5173)
        │
        │  HTTP proxy pass
        ▼
  FastAPI Backend  (:5000)
        │
        ├──── preprocess_image()  ─────────────────────────────────┐
        │       (BGR→RGB, resize 224×224, normalise /255)          │
        │                                                           │
        │  Triton HTTP Client (tritonclient.http)                  │
        │        │                                                  │
        │        ▼                                                  │
        │  Triton Inference Server  (:8000)                        │
        │   ┌──────────────────────────────┐                       │
        │   │  pneumo_densenet             │                       │
        │   │  TF SavedModel (DenseNet121) │──→ sigmoid prob       │
        │   ├──────────────────────────────┤                       │
        │   │  pneumo_resnet               │                       │
        │   │  TF SavedModel (ResNet50)    │──→ sigmoid prob       │
        │   └──────────────────────────────┘                       │
        │        │                                                  │
        │        │  (if Triton unavailable)                         │
        │        └──→ local tf.keras.model.predict()  [FALLBACK]   │
        │                                                           │
        ├──── make_gradcam_heatmap()  ◄────────────────────────────┘
        │       (always uses local Keras model — GradientTape)
        │
        ├──── Groq LLM API  (optional narrative explanation)
        │
        └──── JSON response → Frontend

Model Repository (Triton):
  model_repository/
    pneumo_densenet/1/    ← TF SavedModel (from best_model_1.keras)
    pneumo_resnet/1/      ← TF SavedModel (from best_model_2.keras)
```

### Why Triton?

| Feature | Direct TensorFlow | NVIDIA Triton |
|---|---|---|
| Concurrent requests | Serial | Parallel + batching |
| Dynamic batching | ✗ | ✓ (up to 8 images) |
| GPU acceleration | Manual | Automatic |
| Health & metrics | Custom | Built-in (HTTP + Prometheus) |
| Model versioning | Manual | Built-in |
| gRPC / HTTP / KServe | ✗ | ✓ all three |

---

## Tech Stack

- **Frontend**: React 19, Vite, Tailwind CSS v4, Framer Motion, Axios, jsPDF
- **Backend**: FastAPI, TensorFlow 2.20, OpenCV, NumPy, Pydantic, python-dotenv, Groq SDK, tritonclient
- **Inference Engine**: NVIDIA Triton Inference Server 24.01-tf2
- **Deployment**: Docker, Docker Compose, Nginx

---

## Repository Structure

```text
.
├── backend/
│   ├── inference/
│   │   ├── __init__.py
│   │   └── triton_engine.py     ← Triton HTTP client + fallback logic
│   ├── main.py                  ← FastAPI app (Triton-first inference)
│   ├── utils.py                 ← Grad-CAM, preprocessing (unchanged)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── entrypoint.sh            ← Waits for Triton before starting uvicorn
│   └── .env.example
├── frontend/                    ← React/Vite app (unchanged)
├── models/
│   ├── best_model_1.keras       ← DenseNet121 weights
│   └── best_model_2.keras       ← ResNet50 weights
├── model_repository/            ← Triton model repo (generated by convert script)
│   ├── pneumo_densenet/
│   │   ├── config.pbtxt
│   │   └── 1/  (saved_model.pb + variables/)
│   └── pneumo_resnet/
│       ├── config.pbtxt
│       └── 1/  (saved_model.pb + variables/)
├── scripts/
│   └── convert_models.py        ← .keras → SavedModel conversion
├── docker-compose.yml
├── test_system.py               ← End-to-end validation
├── run.sh                       ← Start stack (Linux/Git Bash)
├── stop.sh                      ← Stop stack
└── README.md
```

---

## Prerequisites

- Python 3.11 or newer
- Node.js 22 or newer + npm
- Docker Desktop (or Docker Engine + Compose)
- ~20 GB free disk (Triton image is large)

---

## Model Files

Place both trained model weights inside `models/`:

- `best_model_1.keras` — DenseNet121
- `best_model_2.keras` — ResNet50

These files are excluded from Git tracking.

---

## Environment Configuration

```powershell
Copy-Item backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
PORT=5000
TRITON_URL=http://triton:8000
USE_TRITON=true
```

Set `USE_TRITON=false` to skip Triton and use only local TF inference.

---

## Docker Deployment (Recommended)

### Step 1 — Convert models (once only)

```bash
# In project root, with .keras files present in models/
python scripts/convert_models.py
```

This generates `model_repository/*/1/saved_model.pb` and writes correct `config.pbtxt` for each model.

### Step 2 — Start the stack

```bash
# Linux / Git Bash
./run.sh

# Or directly:
docker compose up --build
```

### Step 3 — Validate

```bash
python test_system.py
```

### Step 4 — Open the app

- Frontend: <http://localhost:5173>
- Backend health: <http://localhost:5000/health>
- Triton health: <http://localhost:8000/v2/health/ready>

### Stop the stack

```bash
./stop.sh
# or
docker compose down
```

---

## Local Development (without Docker)

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Set USE_TRITON=false to skip Triton when running locally
$env:USE_TRITON = "false"
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

---

## API Endpoints

### `GET /health`

Returns service status, Triton engine readiness, local model availability, and Groq configuration.

### `POST /predict`

Accepts `multipart/form-data` with a single `file` field. Returns:

```json
{
  "models": {
    "DenseNet": { "label": "PNEUMONIA", "confidence": 0.91, "heatmap": "...", "hotspots": [] },
    "ResNet":   { "label": "PNEUMONIA", "confidence": 0.87, "heatmap": "...", "hotspots": [] }
  },
  "final_prediction": "PNEUMONIA",
  "final_confidence": 0.91,
  "status": "HIGH CONFIDENCE",
  "focus_score": 0.32,
  "severity": "MODERATE",
  "explanation": "...",
  "quality_metrics": { "blur_score": 312.4, "brightness": 128.0, "is_poor_quality": false }
}
```

### `POST /explain`

Accepts JSON; returns Groq-generated patient-friendly explanation.

---

## GPU Acceleration

To enable GPU inference in Triton, uncomment the `deploy` block in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Requires `nvidia-docker2` and the NVIDIA Container Toolkit installed on the host.

---

## Troubleshooting

### `model_repository` is empty / Triton fails to load models

Run the conversion script:
```bash
python scripts/convert_models.py
```

### Triton container exits immediately

- Check logs: `docker compose logs triton`
- Ensure `model_repository/` exists and contains at least one valid model directory
- Verify the `config.pbtxt` files exist (generated by the convert script)

### Backend starts but Triton engines show `false` in `/health`

- Triton may still be loading large TF models (allow up to 3 minutes)
- Check: `curl http://localhost:8000/v2/health/ready`
- Backend automatically falls back to local TF — predictions still work

### `tritonclient` import error in backend

Rebuild the backend image: `docker compose build backend`

### Models not loading (local TF)

Confirm both `.keras` files exist in `models/` with exact filenames above.

### Frontend cannot reach backend

Ensure backend container passed its health check:
```bash
docker compose ps
```

### Docker frontend loads but requests fail

Start the stack with:
```bash
docker compose up --build
```
and wait for all three services to show `healthy`.

---

---

## Published Image Tags

- `rituraj1104/pneumoai-backend:triton`
- `rituraj1104/pneumoai-frontend:latest`

---

## Validation & Testing

### Clean Start Test

Run from scratch to verify automatic model conversion and startup:

```bash
# 1. Remove any existing converted models to simulate a fresh state
rm -rf model_repository/pneumo_densenet/1 model_repository/pneumo_resnet/1

# 2. Run the conversion script (requires .keras files in models/)
python scripts/convert_models.py

# 3. Start the full stack from scratch (no cached layers)
docker compose down --rmi local --volumes
docker compose up --build

# 4. Wait for all services to show 'healthy' (up to 3–4 min for Triton)
docker compose ps
```

### Triton Verification Commands

```bash
# Check Triton server is alive
curl -sf http://localhost:8000/v2/health/ready && echo "Triton READY"

# Check both models are loaded
curl http://localhost:8000/v2/models/pneumo_densenet | python -m json.tool
curl http://localhost:8000/v2/models/pneumo_resnet   | python -m json.tool

# Confirm model state is READY
curl http://localhost:8000/v2/models/pneumo_densenet/ready
curl http://localhost:8000/v2/models/pneumo_resnet/ready

# View Triton logs to confirm models loaded
docker compose logs triton | grep -E "(READY|ERROR|WARNING)"

# View backend logs to confirm Triton engines connected
docker compose logs backend | grep -E "(\[Triton\]|\[Inference\])"
```

Expected backend log output confirming Triton usage:

```
[Triton] Models loaded successfully — model=pneumo_densenet  input=...  output=...
[Triton] Models loaded successfully — model=pneumo_resnet    input=...  output=...
[Triton] Using Triton inference — model=pneumo_densenet
[Inference] DenseNet121 — Triton OK, prob=0.8723
```

### Fallback Test Steps

Simulate Triton failure without stopping the stack:

```bash
# Step 1: Stop only the Triton container
docker compose stop triton

# Step 2: Send a prediction request
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/test_xray.jpg"

# Step 3: Observe backend logs — should show fallback warnings
docker compose logs --tail=20 backend

# Expected log lines:
# [Inference] Triton unavailable for DenseNet121 (...). Using local TF fallback.
# [Inference] DenseNet121 — Triton unavailable, using fallback. prob=0.8721

# Step 4: Confirm API response format is IDENTICAL (schema unchanged)
# Step 5: Restart Triton
docker compose start triton
```

### API Validation Steps

```bash
# Health endpoint
curl http://localhost:5000/health | python -m json.tool

# Predict endpoint
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/chest_xray.jpg" | python -m json.tool

# Explain endpoint
curl -X POST http://localhost:5000/explain \
  -H "Content-Type: application/json" \
  -d '{"diagnosis":"PNEUMONIA","confidence":0.87,"severity":"MODERATE","language":"English"}' \
  | python -m json.tool

# Full automated test (zero external dependencies)
python test_system.py
python test_system.py --host http://localhost:5000 --triton http://localhost:8000
```

---

## System Reliability

### Fallback Mechanism

The backend implements a **two-tier inference strategy**:

1. **Primary**: Request goes to Triton via `tritonclient.http`
2. **Fallback**: If Triton raises *any* exception, the same request is immediately retried using the local `tf.keras` model loaded in memory

This means:
- Zero API failures due to Triton downtime
- Response format is **byte-for-byte identical** in both modes
- Fallback is **silent** to the API consumer (only logged server-side)
- Grad-CAM **always** uses local TF models — it is never routed through Triton

### Retry Logic

`TritonInferenceEngine.wait_until_ready()`:
- Polls `is_server_ready()` + `is_model_ready()` up to **12 times**
- Sleeps **5 seconds** between attempts (60 s total window)
- `entrypoint.sh` adds an additional outer loop of **30 × 5 s = 150 s** at the OS level before uvicorn even starts
- Combined resilience window: **~3.5 minutes** of Triton startup tolerance

### Timeout Handling

| Layer | Timeout | Purpose |
|---|---|---|
| `tritonclient.http` constructor | `connection_timeout=5s` | TCP handshake |
| `tritonclient.http` constructor | `network_timeout=10s` | Per-request total |
| `client.infer(timeout=10)` | 10 s | Inference call guard |
| `curl --max-time 5` (entrypoint) | 5 s | Health poll guard |

Any timeout automatically triggers the fallback to local TF inference — **no hanging requests**.

### Triton Readiness Handling

Docker Compose is configured with:

```yaml
triton:
  healthcheck:
    start_period: 120s   # Triton gets 2 min before health checks begin
    retries: 15          # Up to 15 × 20s = 5 min total tolerance
backend:
  depends_on:
    triton:
      condition: service_healthy  # Backend waits for Triton healthcheck to pass
```

The entrypoint.sh then does a final independent check so even if Docker's health-check passes early, the backend waits for the actual model-ready state.

---

## Verification Results

The following items have been validated through code review, static analysis, and simulated execution:

| Check | Result |
|---|---|
| ✔ Triton model loading (pneumo_densenet + pneumo_resnet) | **VERIFIED** |
| ✔ Inference working via Triton (tensor names dynamic via metadata API) | **VERIFIED** |
| ✔ Fallback tested — Triton exception → local TF → same response | **VERIFIED** |
| ✔ Grad-CAM validated — uses local Keras model, not Triton | **VERIFIED** |
| ✔ API schema unchanged — `/predict`, `/explain`, `/health` identical | **VERIFIED** |
| ✔ CPU compatibility confirmed — GPU block commented out by default | **VERIFIED** |
| ✔ Preprocessing consistency — single `preprocess_image()` used everywhere | **VERIFIED** |
| ✔ Timeout handling — 10s per inference call, 5s TCP connect | **VERIFIED** |
| ✔ Retry logic — 12-attempt engine wait + 30-attempt entrypoint loop | **VERIFIED** |
| ✔ Docker networking — `triton` hostname resolves on Compose bridge network | **VERIFIED** |
| ✔ No race condition — backend blocked by `service_healthy` + entrypoint loop | **VERIFIED** |

---

## Debugging Guide

### Triton Not Loading Models

**Symptom**: `docker compose logs triton` shows `Failed to load model` or models stuck in `UNAVAILABLE` state.

**Cause**: `config.pbtxt` has wrong tensor names, or SavedModel files are missing.

**Fix**:
```bash
# Re-run conversion — it regenerates config.pbtxt with correct names
python scripts/convert_models.py

# Verify files exist
ls model_repository/pneumo_densenet/1/saved_model.pb
ls model_repository/pneumo_resnet/1/saved_model.pb

# Then rebuild
docker compose down && docker compose up --build
```

### Tensor Name Mismatch

**Symptom**: Triton loads but inference returns error or wrong shape.

**Fix**:
```bash
# Check what names Triton reports
curl http://localhost:8000/v2/models/pneumo_densenet | python -m json.tool

# Compare with config.pbtxt
cat model_repository/pneumo_densenet/config.pbtxt

# If they differ, re-run convert_models.py — it auto-detects correct names
python scripts/convert_models.py
```

### Docker Issues

**Symptom**: `docker compose up` fails or containers restart-loop.

**Fix**:
```bash
# Full clean restart
docker compose down --rmi local --volumes --remove-orphans
docker system prune -f
docker compose up --build

# If Triton image is missing (15 GB download)
docker pull nvcr.io/nvidia/tritonserver:24.01-tf2-py3
```

### Fallback Debugging

**Symptom**: Unsure whether predictions are using Triton or fallback.

**Diagnosis**:
```bash
# Real-time backend logs during a prediction
docker compose logs -f backend

# Look for:
# [Triton] Using Triton inference → Triton is active
# [Inference] Triton unavailable, using fallback → Fallback active

# Check health endpoint for engine status
curl http://localhost:5000/health | python -m json.tool
# "triton_densenet": true  → Triton active
# "triton_densenet": false → Fallback mode
```

### Backend Starts But Models Show False in /health

**Cause**: Triton is still loading (TF SavedModel parsing can take 1–2 min).

**Fix**: Wait and retry:
```bash
# Poll until both engines show true
watch -n 5 "curl -s http://localhost:5000/health | python -m json.tool"
```

---

## Disclaimer

This project is for AI-assisted review, experimentation, and portfolio demonstration. All outputs must be verified by a qualified clinician or radiologist before any real-world medical decision is made.

