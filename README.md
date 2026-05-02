# PneumoAI: Pneumonia Detection Using Deep Learning

PneumoAI is a full-stack chest X-ray screening workspace built around a dual-model deep learning ensemble. It combines real image inference, Grad-CAM explainability, image quality checks, AI-generated narrative support, and exportable clinical reporting in a single deployable project.

> This repository is intended for educational and research use only. It is not a medical device and must not be used as a substitute for licensed clinical judgment.

## Highlights

- Dual-model inference using DenseNet121 and ResNet50
- Real FastAPI backend with TensorFlow model loading
- React + Vite clinical review UI
- Grad-CAM heatmap inspection with compare, overlay, split, and original views
- Image quality assessment before interpretation
- Groq-powered clinical explanation endpoint with graceful fallback
- Export-ready PDF clinical report generation
- Local development and Docker Compose deployment support

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

## System Overview

For each uploaded chest X-ray, the application:

1. validates the file type and size in the frontend
2. preprocesses the image for the TensorFlow models
3. runs DenseNet121 and ResNet50 inference
4. fuses both outputs into a final prediction and confidence score
5. generates Grad-CAM heatmaps and hotspot metadata
6. calculates image quality metrics and an attention-derived severity estimate
7. optionally requests a narrative explanation from Groq
8. allows the user to export an on-screen result as a PDF clinical report

## Architecture

```text
Frontend (React + Vite)
  -> /predict, /explain, /health
FastAPI backend
  -> TensorFlow model loading
  -> preprocessing + quality checks
  -> Grad-CAM generation
  -> Groq explanation service
Models
  -> best_model_1.keras
  -> best_model_2.keras
```

## Tech Stack

- Frontend: React 19, Vite, Tailwind CSS v4, Framer Motion, Axios, jsPDF
- Backend: FastAPI, TensorFlow 2.20, OpenCV, NumPy, Pydantic, python-dotenv, Groq SDK
- Deployment: Docker, Docker Compose, Nginx

## Repository Structure

```text
.
|- backend/
|  |- main.py
|  |- utils.py
|  |- requirements.txt
|  |- Dockerfile
|  |- .env.example
|- frontend/
|  |- src/
|  |- public/
|  |- Dockerfile
|  |- nginx.conf
|  |- package.json
|- models/
|  |- README.md
|- screenshots/
|- docker-compose.yml
|- package.json
|- .env.example
|- README.md
```

## Prerequisites

- Python 3.11 or newer recommended
- Node.js 22 or newer recommended
- npm
- Docker Desktop or Docker Engine with Compose support for containerized runs

## Model Files

The backend expects the trained weights below inside `models/`:

- `best_model_1.keras`
- `best_model_2.keras`

See [`models/README.md`](./models/README.md) for the expected filenames.

These model files are intentionally excluded from Git tracking.

## Environment Configuration

Copy the backend example environment file:

```powershell
Copy-Item backend/.env.example backend/.env
```

Minimum expected values:

```env
GROQ_API_KEY=your_groq_api_key_here
PORT=5000
```

Notes:

- `GROQ_API_KEY` is optional for basic prediction.
- If the Groq key is missing, `/explain` returns a fallback message instead of a live generated narrative.
- The backend loads variables from both the project root `.env` and `backend/.env`, but `backend/.env` is the recommended location for this repository.

## Local Development

### 1. Backend

Create a virtual environment and install backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Run the API:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

Health endpoint:

```text
http://localhost:5000/health
```

### 2. Frontend

Install frontend dependencies and start the Vite app:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

The Vite development server already proxies `/predict`, `/explain`, and `/health` to `http://localhost:5000`.

## Root Helper Scripts

From the repository root:

```powershell
npm run frontend:dev
npm run frontend:build
npm run backend:dev
npm run docker:up
npm run docker:down
```

## Docker Deployment

This repository already includes:

- `backend/Dockerfile`
- `frontend/Dockerfile`
- `frontend/nginx.conf`
- `docker-compose.yml`

### Build and start

```powershell
docker compose up --build
```

### Services

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:5000`

The frontend container is served by Nginx and proxies API traffic to the backend container internally.

### Stop containers

```powershell
docker compose down
```

## API Endpoints

### `GET /health`

Returns:

- service status
- model availability for DenseNet and ResNet
- whether Groq is configured

### `POST /predict`

Accepts `multipart/form-data` with a single `file` field and returns:

- final prediction
- final confidence
- ensemble status
- severity and focus score
- per-model confidence
- Grad-CAM images and hotspot metadata
- quality metrics
- backend summary text

### `POST /explain`

Accepts JSON like:

```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 0.92,
  "severity": "MODERATE",
  "language": "English"
}
```

Returns a patient-friendly, paragraph-style explanation generated through Groq when configured.

## GitHub Publishing Notes

- Keep `backend/.env` out of version control.
- Do not commit live API keys.
- Keep trained model weights out of version control unless you intentionally change the repository policy.
- Screenshots can be committed safely and are used by this README.
- `run.txt` is intentionally excluded from Git and Docker context.

## Troubleshooting

### Models are not loading

Confirm that both `.keras` files exist in `models/` and match the exact filenames documented above.

### The frontend cannot analyze images

Check that the backend is running and that `http://localhost:5000/health` returns a healthy response.

### Detailed explanation is unavailable

Verify that `backend/.env` contains a valid `GROQ_API_KEY`, then restart the backend container or local server.

### Docker frontend loads but requests fail

Confirm that the backend container passed its health check and that the stack was started with:

```powershell
docker compose up --build
```

## Disclaimer

This project is for AI-assisted review, experimentation, and portfolio demonstration. All outputs must be verified by a qualified clinician or radiologist before any real-world medical decision is made.
