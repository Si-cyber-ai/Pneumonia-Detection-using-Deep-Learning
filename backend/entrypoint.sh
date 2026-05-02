#!/usr/bin/env bash
# =============================================================================
# backend/entrypoint.sh
# =============================================================================
# Waits for Triton Inference Server to become ready before launching uvicorn.
# Eliminates race conditions in Docker Compose startup ordering.
#
# Behaviour:
#   - If USE_TRITON=false/0/no → skip wait entirely
#   - If Triton never becomes ready within MAX_RETRIES × SLEEP_SECS seconds
#     → start backend anyway; Python fallback to local TF will activate
#   - On success → log confirmation and launch uvicorn
# =============================================================================

# Use -uo instead of -euo so that a failing curl in the until-loop does NOT
# abort the script before the loop body can count the attempt and break.
set -uo pipefail

TRITON_HOST="${TRITON_URL:-http://triton:8000}"
USE_TRITON="${USE_TRITON:-true}"
MAX_RETRIES=30
SLEEP_SECS=5

# Strip trailing slash to avoid double-slash in URL construction
TRITON_HOST="${TRITON_HOST%/}"

log() { echo "[entrypoint] $(date -u '+%H:%M:%S') $*"; }

# ---- Triton readiness wait loop --------------------------------------------
if [ "$USE_TRITON" = "true" ] || [ "$USE_TRITON" = "1" ]; then
    log "Waiting for Triton at ${TRITON_HOST}/v2/health/ready ..."
    attempt=0
    triton_ready=false

    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        # curl -sf: -s silent, -f fail-on-HTTP-error; exit code 0 = success
        if curl -sf --max-time 5 "${TRITON_HOST}/v2/health/ready" > /dev/null 2>&1; then
            triton_ready=true
            break
        fi
        attempt=$((attempt + 1))
        log "  Triton not ready (attempt ${attempt}/${MAX_RETRIES}). Retrying in ${SLEEP_SECS}s ..."
        sleep "$SLEEP_SECS"
    done

    if [ "$triton_ready" = "true" ]; then
        log "Triton is ready. Proceeding to start backend."
    else
        log "WARNING: Triton did not become ready after ${MAX_RETRIES} attempts ($(( MAX_RETRIES * SLEEP_SECS ))s)."
        log "Starting backend anyway — local TF inference fallback will be used."
    fi
else
    log "USE_TRITON='${USE_TRITON}' — skipping Triton wait. Local TF inference only."
fi

# ---- Start FastAPI ----------------------------------------------------------
log "Starting PneumoAI FastAPI backend on port ${PORT:-5000} ..."
exec python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --workers 1
