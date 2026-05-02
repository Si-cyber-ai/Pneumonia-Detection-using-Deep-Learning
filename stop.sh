#!/usr/bin/env bash
# =============================================================================
# stop.sh — Gracefully stop and remove the PneumoAI Docker Compose stack
# Compatible with: Linux, macOS, Git Bash on Windows
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  PneumoAI — Stopping Triton Edition stack"
echo "============================================================"

docker compose down

echo ""
echo "[OK] All containers stopped."
echo ""
echo "  To also remove images:"
echo "    docker compose down --rmi local"
echo ""
echo "  To remove volumes:"
echo "    docker compose down -v"
echo "============================================================"
