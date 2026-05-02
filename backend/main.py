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

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(APP_DIR / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH_1 = PROJECT_ROOT / "models" / "best_model_1.keras"
MODEL_PATH_2 = PROJECT_ROOT / "models" / "best_model_2.keras"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "")


def load_model_if_available(model_path: Path, model_name: str):
    if not model_path.exists():
        logger.error("%s model not found at %s.", model_name, model_path)
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        logger.exception("Failed to load %s model from %s.", model_name, model_path)
        return None


densenet_model = load_model_if_available(MODEL_PATH_1, "DenseNet121")
resnet_model = load_model_if_available(MODEL_PATH_2, "ResNet50")


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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "DenseNet": densenet_model is not None,
            "ResNet": resnet_model is not None,
        },
        "groq_configured": bool(GROQ_API_KEY),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if densenet_model is None or resnet_model is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction models are not loaded. Real analysis is unavailable until both model files load successfully.",
            )

        image_bytes = await file.read()
        img_tensor, resized_img = preprocess_image(image_bytes)
        quality_metrics = check_image_quality(resized_img)

        dense_pred = float(densenet_model.predict(img_tensor, verbose=0)[0][0])
        res_pred = float(resnet_model.predict(img_tensor, verbose=0)[0][0])

        dense_label = format_prediction(dense_pred)
        res_label = format_prediction(res_pred)

        dense_heatmap = make_gradcam_heatmap(img_tensor, densenet_model)
        res_heatmap = make_gradcam_heatmap(img_tensor, resnet_model)

        dense_confidence = float(dense_pred if dense_label == "PNEUMONIA" else 1.0 - dense_pred)
        res_confidence = float(res_pred if res_label == "PNEUMONIA" else 1.0 - res_pred)

        if dense_label == res_label:
            status = "HIGH CONFIDENCE"
            final_label = dense_label
            final_conf = max(dense_confidence, res_confidence)
        else:
            status = "REVIEW REQUIRED"
            if dense_confidence > res_confidence:
                final_label = dense_label
                final_conf = dense_confidence
            else:
                final_label = res_label
                final_conf = res_confidence

        dense_cam_b64, dense_mask_b64, orig_b64 = generate_gradcam_base64(resized_img, dense_heatmap)
        res_cam_b64, res_mask_b64, _ = generate_gradcam_base64(resized_img, res_heatmap)
        dense_hotspots = extract_hotspots(dense_heatmap)
        res_hotspots = extract_hotspots(res_heatmap)

        focus = calculate_focus_score(dense_heatmap)
        severity = get_severity(focus)
        explanation = build_prediction_summary(final_label, status, severity)

        return {
            "models": {
                "DenseNet": {
                    "label": dense_label,
                    "confidence": round(dense_confidence, 4),
                    "heatmap": dense_cam_b64,
                    "heatmap_mask": dense_mask_b64,
                    "original": orig_b64,
                    "hotspots": dense_hotspots,
                },
                "ResNet": {
                    "label": res_label,
                    "confidence": round(res_confidence, 4),
                    "heatmap": res_cam_b64,
                    "heatmap_mask": res_mask_b64,
                    "original": orig_b64,
                    "hotspots": res_hotspots,
                },
            },
            "final_prediction": final_label,
            "final_confidence": round(final_conf, 4),
            "status": status,
            "focus_score": round(focus, 4),
            "severity": severity,
            "explanation": explanation,
            "quality_metrics": quality_metrics,
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


class ExplainRequest(BaseModel):
    diagnosis: str
    confidence: float
    severity: str
    language: str = "English"


@app.post("/explain")
async def explain_diagnosis(req: ExplainRequest):
    if not GROQ_API_KEY:
        return {"explanation": "Groq API key is not configured. Set GROQ_API_KEY or LLM_API_KEY for live explanations."}

    prompt = (
        f"The patient's chest X-ray was analyzed by an AI model and showed a diagnosis of '{req.diagnosis}' "
        f"with a confidence of {req.confidence * 100}%. The severity is estimated as '{req.severity}'. "
        f"Please provide a professional patient-facing clinical explanation of what this means, what the next steps should be, "
        f"and general advice for managing this condition. Return exactly 4 to 5 well-structured paragraphs, with no headings, "
        f"no bullet points, no numbering, no markdown, and no special formatting characters. Output only the translated explanation "
        f"in {req.language}. If the requested language is not English, do not include English text."
    )

    models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    max_retries = 3

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
                            "content": "You are a professional medical AI assistant. Respond with concise, clinically appropriate, patient-friendly paragraphs only. Never use headings, lists, markdown, or bullet points.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    model=model_name,
                    temperature=0.3,
                    max_tokens=1024,
                )
                return {"explanation": response.choices[0].message.content}
            except Exception as exc:
                logger.warning("Error calling %s on attempt %s: %s", model_name, attempt + 1, exc)
                continue

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return {"explanation": "The AI service is currently unavailable. Please try again later."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
