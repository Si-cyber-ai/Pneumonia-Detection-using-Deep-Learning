#!/usr/bin/env python3
"""
PneumoAI Model Conversion Script
Converts .keras models to TensorFlow SavedModel format for NVIDIA Triton Inference Server.

Run this ONCE before starting Docker Compose:
    python scripts/convert_models.py

Requirements:
    pip install tensorflow==2.20.0
"""

import json
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_REPO   = PROJECT_ROOT / "model_repository"

MODELS = [
    {
        "keras_file":   "best_model_1.keras",
        "triton_name":  "pneumo_densenet",
        "display_name": "DenseNet121",
    },
    {
        "keras_file":   "best_model_2.keras",
        "triton_name":  "pneumo_resnet",
        "display_name": "ResNet50",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inspect_savedmodel(version_dir: Path):
    """Extract input/output tensor names from SavedModel serving_default signature."""
    import tensorflow as tf

    loaded = tf.saved_model.load(str(version_dir))

    if "serving_default" not in loaded.signatures:
        raise RuntimeError(
            "No 'serving_default' signature found. Available: "
            + str(list(loaded.signatures.keys()))
        )

    sig = loaded.signatures["serving_default"]

    # structured_input_signature → (positional_list, keyword_dict)
    _, input_specs = sig.structured_input_signature
    input_name = list(input_specs.keys())[0]

    # structured_outputs → {name: tensor}
    output_name = list(sig.structured_outputs.keys())[0]

    return input_name, output_name


def write_config_pbtxt(
    model_dir: Path,
    model_name: str,
    input_name: str,
    output_name: str,
    max_batch_size: int = 8,
):
    """Generate Triton config.pbtxt with correct tensor names."""
    config = (
        f'name: "{model_name}"\n'
        f'backend: "tensorflow"\n'
        f'max_batch_size: {max_batch_size}\n'
        f'\n'
        f'input [\n'
        f'  {{\n'
        f'    name: "{input_name}"\n'
        f'    data_type: TYPE_FP32\n'
        f'    dims: [224, 224, 3]\n'
        f'  }}\n'
        f']\n'
        f'\n'
        f'output [\n'
        f'  {{\n'
        f'    name: "{output_name}"\n'
        f'    data_type: TYPE_FP32\n'
        f'    dims: [1]\n'
        f'  }}\n'
        f']\n'
        f'\n'
        f'dynamic_batching {{\n'
        f'  preferred_batch_size: [1, 2, 4, 8]\n'
        f'  max_queue_delay_microseconds: 100\n'
        f'}}\n'
    )
    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config)
    print(f"  [OK] config.pbtxt  → {config_path}")


def convert_model(keras_file: str, triton_name: str, display_name: str):
    import tensorflow as tf

    keras_path  = MODELS_DIR / keras_file
    model_dir   = MODEL_REPO / triton_name
    version_dir = model_dir / "1"

    print(f"\n{'='*60}")
    print(f"Converting: {display_name}")
    print(f"  Source  : {keras_path}")
    print(f"  Target  : {version_dir}")

    if not keras_path.exists():
        print(f"  [ERROR] File not found: {keras_path}")
        print("  Place both .keras files in the models/ directory and re-run.")
        sys.exit(1)

    # Clean stale version directory
    if version_dir.exists():
        print(f"  Removing existing {version_dir} …")
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    # Load .keras model
    print(f"  Loading {display_name} …")
    model = tf.keras.models.load_model(str(keras_path))
    print(f"  Model loaded. Input shape: {model.input_shape}")

    # Save as SavedModel directly into version_dir (saved_model.pb + variables/)
    print(f"  Saving as TF SavedModel …")
    tf.saved_model.save(model, str(version_dir))
    print(f"  SavedModel written.")

    # Inspect to get actual tensor names
    print(f"  Inspecting SavedModel signatures …")
    input_name, output_name = inspect_savedmodel(version_dir)
    print(f"  Input  tensor: {input_name}")
    print(f"  Output tensor: {output_name}")

    # Write config.pbtxt
    write_config_pbtxt(model_dir, triton_name, input_name, output_name)

    # Write a JSON metadata file for the backend to optionally read
    meta = {"input_name": input_name, "output_name": output_name}
    meta_path = model_dir / "tensor_names.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [OK] tensor_names.json → {meta_path}")

    print(f"  [DONE] {display_name} conversion complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("PneumoAI Model Conversion → Triton SavedModel Format")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model repo   : {MODEL_REPO}")

    MODEL_REPO.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf
        print(f"TensorFlow   : {tf.__version__}")
    except ImportError:
        print("[ERROR] TensorFlow not installed. Run: pip install tensorflow==2.20.0")
        sys.exit(1)

    for m in MODELS:
        convert_model(m["keras_file"], m["triton_name"], m["display_name"])

    print(f"\n{'='*60}")
    print("All models converted successfully.")
    print("Model repository is ready for Triton Inference Server.")
    print(f"\nNext step: docker compose up --build")


if __name__ == "__main__":
    main()
