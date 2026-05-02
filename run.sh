#!/usr/bin/env bash
# =============================================================================
# run.sh — Build and start the full PneumoAI + Triton stack
# Compatible with: Linux, macOS, Git Bash on Windows
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  PneumoAI — Triton Edition startup"
echo "============================================================"

# ---- Step 1: Check model repository ----------------------------------------
MODEL_REPO="$PROJECT_ROOT/model_repository"
DENSENET_PB="$MODEL_REPO/pneumo_densenet/1/saved_model.pb"
RESNET_PB="$MODEL_REPO/pneumo_resnet/1/saved_model.pb"

if [ ! -f "$DENSENET_PB" ] || [ ! -f "$RESNET_PB" ]; then
    echo ""
    echo "[INFO] SavedModel files not found. Running model conversion …"
    echo "       (requires TensorFlow and the .keras files in models/)"
    echo ""
    python scripts/convert_models.py
    echo ""
fi

echo "[OK] Model repository ready."

# ---- Step 2: Docker Compose -------------------------------------------------
echo ""
echo "Starting Docker Compose stack …"
docker compose up --build -d

echo ""
echo "============================================================"
echo "  Stack is starting. Services:"
echo "    Triton   → http://localhost:8000"
echo "    Backend  → http://localhost:5000"
echo "    Frontend → http://localhost:5173"
echo ""
echo "  Run validation:"
echo "    python test_system.py"
echo ""
echo "  Watch logs:"
echo "    docker compose logs -f"
echo "============================================================"
