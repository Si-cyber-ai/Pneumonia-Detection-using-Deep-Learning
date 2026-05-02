"""
PneumoAI FastAPI Backend
========================
Inference pipeline:
    /predict → TritonInferenceEngine  → Triton Server (primary)
                                      ↓ (if Triton unavailable)
                 local tf.keras model (fallback — automatic, no crash)

Grad-CAM always uses the local tf.keras models because GradientTape requires
access to intermediate layer activations, which Triton does not expose.

All API endpoints, request formats, and response schemas are UNCHANGED.
"""

import asyncio
import logging
import os
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel

try:
    from utils import (
        preprocess_image,
        check_image_quality,
        make_gradcam_heatmap,
        generate_gradcam_base64,
        calculate_focus_score,
        get_severity,
        extract_hotspots,
    )
    from inference.triton_engine import TritonInferenceEngine
except ImportError:
    from .utils import (
        preprocess_image,
        check_image_quality,
        make_gradcam_heatmap,
        generate_gradcam_base64,
        calculate_focus_score,
        get_severity,
        extract_hotspots,
    )
    from .inference.triton_engine import TritonInferenceEngine

# ---------------------------------------------------------------------------
# Paths & Environment
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(APP_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "")
TRITON_URL   = os.getenv("TRITON_URL", "http://triton:8000")
USE_TRITON   = os.getenv("USE_TRITON", "true").lower() not in ("false", "0", "no")

MODEL_PATH_1 = PROJECT_ROOT / "models" / "best_model_1.keras"
MODEL_PATH_2 = PROJECT_ROOT / "models" / "best_model_2.keras"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PneumoAI",
    description="Pneumonia detection via dual-model deep-learning ensemble with Triton backend.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_keras_model(model_path: Path, model_name: str):
    """Load a .keras model; return None on any failure."""
    if not model_path.exists():
        logger.error("%s: file not found at %s", model_name, model_path)
        return None
    try:
        m = tf.keras.models.load_model(model_path)
        logger.info("%s loaded from %s", model_name, model_path)
        return m
    except Exception:
        logger.exception("Failed to load %s from %s", model_name, model_path)
        return None


def init_triton_engine(model_name: str) -> "TritonInferenceEngine | None":
    """
    Create a TritonInferenceEngine and wait for it to be ready.
    Returns None if Triton is disabled or unreachable.
    """
    if not USE_TRITON:
        logger.info("Triton disabled via USE_TRITON env var. Using local TF only.")
        return None
    try:
        engine = TritonInferenceEngine(TRITON_URL, model_name)
        ready  = engine.wait_until_ready()
        if ready:
            return engine
        return None
    except Exception:
        logger.exception("Could not create TritonInferenceEngine for %s", model_name)
        return None


# ---------------------------------------------------------------------------
# Boot-time initialisation — run at import/startup (synchronous is fine here)
# ---------------------------------------------------------------------------

# Local tf.keras models — ALWAYS loaded; used for Grad-CAM and as fallback
densenet_model = load_keras_model(MODEL_PATH_1, "DenseNet121")
resnet_model   = load_keras_model(MODEL_PATH_2, "ResNet50")

# Triton engines — primary inference path
densenet_engine = init_triton_engine("pneumo_densenet")
resnet_engine   = init_triton_engine("pneumo_resnet")

logger.info(
    "Startup complete — Triton: densenet=%s resnet=%s | Local TF: densenet=%s resnet=%s",
    densenet_engine is not None,
    resnet_engine   is not None,
    densenet_model  is not None,
    resnet_model    is not None,
)

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(engine, local_model, img_tensor, model_label: str) -> float:
    """
    Execute inference via Triton (primary) with automatic fallback to local TF.

    Returns the raw sigmoid probability as a float.
    Raises RuntimeError if neither path is available.
    """
    # --- Triton path ---
    if engine is not None:
        try:
            prob = engine.predict(img_tensor)   # already logs "Using Triton inference"
            logger.info(
                "[Inference] %s — Triton OK, prob=%.4f", model_label, prob
            )
            return prob
        except Exception as exc:
            logger.warning(
                "[Inference] Triton unavailable for %s (%s). Using local TF fallback.",
                model_label, exc,
            )

    # --- Local TF fallback ---
    if local_model is not None:
        prob = float(local_model.predict(img_tensor, verbose=0)[0][0])
        logger.warning(
            "[Inference] %s — Triton unavailable, using fallback. prob=%.4f", model_label, prob
        )
        return prob

    raise RuntimeError(
        f"No inference backend available for {model_label}. "
        "Neither Triton nor local TF model is loaded."
    )



def format_prediction(probability: float) -> str:
    return "PNEUMONIA" if probability > 0.5 else "NORMAL"


def build_prediction_summary(final_label: str, status: str, severity: str) -> str:
    if final_label == "PNEUMONIA":
        return (
            f"The ensemble detected pneumonia-related opacity patterns with status {status.lower()} "
            f"and estimated severity {severity.lower()}."
        )
    return (
        f"The ensemble did not detect pneumonia and returned status {status.lower()} "
        f"with severity {severity.lower()}."
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "inference": {
            "triton_densenet": densenet_engine is not None and densenet_engine.is_ready,
            "triton_resnet":   resnet_engine   is not None and resnet_engine.is_ready,
            "triton_url":      TRITON_URL if USE_TRITON else "disabled",
        },
        "models": {
            "DenseNet": densenet_model is not None,
            "ResNet":   resnet_model   is not None,
        },
        "groq_configured": bool(GROQ_API_KEY),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # At least one complete path (Triton or local TF) must exist per model
        densenet_ok = (densenet_engine is not None) or (densenet_model is not None)
        resnet_ok   = (resnet_engine   is not None) or (resnet_model   is not None)

        if not densenet_ok or not resnet_ok:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Prediction models are not loaded. "
                    "Real analysis is unavailable until both model files load successfully."
                ),
            )

        image_bytes = await file.read()
        img_tensor, resized_img = preprocess_image(image_bytes)
        quality_metrics = check_image_quality(resized_img)

        # --- Primary inference (Triton → local TF fallback) ---
        dense_pred = run_inference(densenet_engine, densenet_model, img_tensor, "DenseNet121")
        res_pred   = run_inference(resnet_engine,   resnet_model,   img_tensor, "ResNet50")

        dense_label = format_prediction(dense_pred)
        res_label   = format_prediction(res_pred)

        # --- Grad-CAM always uses local tf.keras models ---
        # Guard: if local model is None (e.g. file missing), return blank heatmap
        # so inference still succeeds — Grad-CAM is non-critical to the prediction.
        import numpy as np
        blank_heatmap = np.zeros((224, 224), dtype=np.float32)

        if densenet_model is not None:
            dense_heatmap = make_gradcam_heatmap(img_tensor, densenet_model)
        else:
            logger.warning("[GradCAM] DenseNet local model not loaded — returning blank heatmap.")
            dense_heatmap = blank_heatmap

        if resnet_model is not None:
            res_heatmap = make_gradcam_heatmap(img_tensor, resnet_model)
        else:
            logger.warning("[GradCAM] ResNet local model not loaded — returning blank heatmap.")
            res_heatmap = blank_heatmap


        dense_confidence = float(dense_pred if dense_label == "PNEUMONIA" else 1.0 - dense_pred)
        res_confidence   = float(res_pred   if res_label   == "PNEUMONIA" else 1.0 - res_pred)

        if dense_label == res_label:
            status      = "HIGH CONFIDENCE"
            final_label = dense_label
            final_conf  = max(dense_confidence, res_confidence)
        else:
            status = "REVIEW REQUIRED"
            if dense_confidence > res_confidence:
                final_label = dense_label
                final_conf  = dense_confidence
            else:
                final_label = res_label
                final_conf  = res_confidence

        dense_cam_b64, dense_mask_b64, orig_b64 = generate_gradcam_base64(resized_img, dense_heatmap)
        res_cam_b64,   res_mask_b64,   _         = generate_gradcam_base64(resized_img, res_heatmap)

        dense_hotspots = extract_hotspots(dense_heatmap)
        res_hotspots   = extract_hotspots(res_heatmap)

        focus       = calculate_focus_score(dense_heatmap)
        severity    = get_severity(focus)
        explanation = build_prediction_summary(final_label, status, severity)

        # Response schema is IDENTICAL to the original system
        return {
            "models": {
                "DenseNet": {
                    "label":        dense_label,
                    "confidence":   round(dense_confidence, 4),
                    "heatmap":      dense_cam_b64,
                    "heatmap_mask": dense_mask_b64,
                    "original":     orig_b64,
                    "hotspots":     dense_hotspots,
                },
                "ResNet": {
                    "label":        res_label,
                    "confidence":   round(res_confidence, 4),
                    "heatmap":      res_cam_b64,
                    "heatmap_mask": res_mask_b64,
                    "original":     orig_b64,
                    "hotspots":     res_hotspots,
                },
            },
            "final_prediction": final_label,
            "final_confidence": round(final_conf, 4),
            "status":           status,
            "focus_score":      round(focus, 4),
            "severity":         severity,
            "explanation":      explanation,
            "quality_metrics":  quality_metrics,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed while processing %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed while processing the uploaded image.",
        ) from exc


# ---------------------------------------------------------------------------
# /explain endpoint — UNCHANGED
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    diagnosis:  str
    confidence: float
    severity:   str
    language:   str = "English"


@app.post("/explain")
async def explain_diagnosis(req: ExplainRequest):
    if not GROQ_API_KEY:
        return {
            "explanation": (
                "Groq API key is not configured. "
                "Set GROQ_API_KEY or LLM_API_KEY for live explanations."
            )
        }

    prompt = (
        f"The patient's chest X-ray was analyzed by an AI model and showed a diagnosis of "
        f"'{req.diagnosis}' with a confidence of {req.confidence * 100}%. "
        f"The severity is estimated as '{req.severity}'. "
        f"Please provide a professional patient-facing clinical explanation of what this means, "
        f"what the next steps should be, and general advice for managing this condition. "
        f"Return exactly 4 to 5 well-structured paragraphs, with no headings, no bullet points, "
        f"no numbering, no markdown, and no special formatting characters. "
        f"Output only the translated explanation in {req.language}. "
        f"If the requested language is not English, do not include English text."
    )

    models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    max_retries   = 3

    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
    except Exception:
        logger.exception("Error initializing Groq client")
        return {"explanation": "An error occurred while initializing the AI service. Please try again later."}

    for attempt in range(max_retries):
        for model_name in models_to_try:
            try:
                response = await client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional medical AI assistant. Respond with concise, "
                                "clinically appropriate, patient-friendly paragraphs only. "
                                "Never use headings, lists, markdown, or bullet points."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=model_name,
                    temperature=0.3,
                    max_tokens=1024,
                )
                return {"explanation": response.choices[0].message.content}
            except Exception as exc:
                logger.warning(
                    "Error calling %s on attempt %d: %s", model_name, attempt + 1, exc
                )
                continue

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return {"explanation": "The AI service is currently unavailable. Please try again later."}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
