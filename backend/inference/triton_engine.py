"""
TritonInferenceEngine
─────────────────────
Wraps tritonclient.http to send inference requests to NVIDIA Triton Inference
Server.  Raises on failure so the caller in main.py can transparently fall back
to local TensorFlow inference.

Design decisions
-----------------
* Input/output tensor names are fetched DYNAMICALLY from Triton's metadata
  endpoint via `as_json=True`, which returns a plain dict. This avoids all
  hardcoding and auto-adapts to whatever names the SavedModel exports.
* Batch dimension (axis 0) is handled by the client; config.pbtxt dims exclude it.
* Per-request timeout of 10 s prevents hangs on an overloaded Triton.
* `wait_until_ready()` uses a bounded while-loop with linear back-off so Docker
  Compose startup is race-condition-free.
* All "using Triton" / "using fallback" events are logged at INFO/WARNING level
  so they are always visible in production log streams.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TritonInferenceEngine:
    """
    Thin client around tritonclient.http for a single Triton model.

    Parameters
    ----------
    triton_url  : base HTTP URL of the Triton server, e.g. "http://triton:8000"
    model_name  : name declared in config.pbtxt, e.g. "pneumo_densenet"
    """

    def __init__(self, triton_url: str, model_name: str) -> None:
        self.triton_url  = triton_url.rstrip("/")
        self.model_name  = model_name
        self.client      = None          # tritonclient.http.InferenceServerClient
        self.input_name: Optional[str]  = None
        self.output_name: Optional[str] = None
        self._ready      = False

        self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Create the HTTP client object (no network call yet)."""
        try:
            import tritonclient.http as httpclient
            # URL must NOT include the http:// scheme for tritonclient
            url_no_scheme = self.triton_url.replace("https://", "").replace("http://", "")
            self.client = httpclient.InferenceServerClient(
                url=url_no_scheme,
                verbose=False,
                connection_timeout=5.0,   # TCP connect timeout
                network_timeout=10.0,     # Per-request network timeout
            )
            logger.info(
                "[Triton] Client created — model=%s  url=%s",
                self.model_name, self.triton_url,
            )
        except ImportError as exc:
            raise RuntimeError(
                "tritonclient[http] is not installed. "
                "Rebuild the Docker image after updating requirements.txt."
            ) from exc

    # ------------------------------------------------------------------
    # Readiness polling
    # ------------------------------------------------------------------

    def wait_until_ready(
        self,
        max_retries: int = 12,
        sleep_secs:  float = 5.0,
    ) -> bool:
        """
        Poll Triton's server-ready and model-ready endpoints.

        Returns True if Triton is ready within the retry window, False otherwise.
        Does NOT raise — callers decide whether to abort or fall back.
        """
        for attempt in range(1, max_retries + 1):
            try:
                server_ok = self.client.is_server_ready()
                model_ok  = self.client.is_model_ready(self.model_name)
                if server_ok and model_ok:
                    self._fetch_metadata()
                    self._ready = True
                    logger.info(
                        "[Triton] Models loaded successfully — model=%s  input=%s  output=%s",
                        self.model_name, self.input_name, self.output_name,
                    )
                    return True
                logger.warning(
                    "[Triton] Not ready yet (attempt %d/%d) — server=%s  model=%s",
                    attempt, max_retries, server_ok, model_ok,
                )
            except Exception as exc:
                logger.warning(
                    "[Triton] Connection attempt %d/%d failed: %s",
                    attempt, max_retries, exc,
                )

            # Don't sleep after the last attempt — return immediately
            if attempt < max_retries:
                time.sleep(sleep_secs)

        logger.error(
            "[Triton] Did not become ready after %d attempts. "
            "Backend will use local TensorFlow fallback.",
            max_retries,
        )
        return False

    # ------------------------------------------------------------------
    # Metadata discovery
    # ------------------------------------------------------------------

    def _fetch_metadata(self) -> None:
        """
        Query Triton's model metadata endpoint and extract input/output names.

        IMPORTANT: tritonclient.http.get_model_metadata() returns a ModelMetadataResponse
        object, NOT a plain dict. The `as_json=True` parameter causes it to return
        a dict, which can be subscripted with ["inputs"] / ["outputs"].
        """
        meta = self.client.get_model_metadata(self.model_name, as_json=True)
        # meta shape: {"name": "...", "inputs": [{"name": ..., "datatype": ...}], "outputs": [...]}
        self.input_name  = meta["inputs"][0]["name"]
        self.output_name = meta["outputs"][0]["name"]
        logger.info(
            "[Triton] Tensor names resolved — input='%s'  output='%s'",
            self.input_name, self.output_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, img_tensor: np.ndarray) -> float:
        """
        Send a preprocessed image tensor to Triton and return the sigmoid probability.

        Parameters
        ----------
        img_tensor : np.ndarray  shape (1, 224, 224, 3), dtype float32
                     Must already be normalised to [0, 1] by preprocess_image().

        Returns
        -------
        float in [0, 1] — raw sigmoid output (same as tf.keras.model.predict()[0][0]).

        Raises
        ------
        RuntimeError  if engine is not ready or Triton returns an error.
        """
        if not self._ready:
            raise RuntimeError(
                f"[Triton] Engine for '{self.model_name}' is not ready. "
                "Caller should fall back to local TF inference."
            )

        import tritonclient.http as httpclient

        tensor = img_tensor.astype(np.float32)
        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, axis=0)   # (224,224,3) → (1,224,224,3)

        # Build Triton InferInput with the exact shape and dtype
        infer_input = httpclient.InferInput(
            self.input_name,
            list(tensor.shape),
            "FP32",
        )
        infer_input.set_data_from_numpy(tensor)

        infer_output = httpclient.InferRequestedOutput(self.output_name)

        logger.info("[Triton] Using Triton inference — model=%s", self.model_name)

        response = self.client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output],
            timeout=10,          # seconds — prevents hanging on slow Triton
        )

        result = response.as_numpy(self.output_name)   # shape: (1, 1)
        prob   = float(result[0][0])
        logger.info(
            "[Triton] Inference complete — model=%s  prob=%.4f",
            self.model_name, prob,
        )
        return prob
