#!/usr/bin/env bash
# One-command setup: installs uv if missing, then syncs the project venv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python pin 3.11 --quiet || true
uv sync --extra dev

echo ""
echo "Setup complete. Try:"
echo "  uv run robosim scenes"
echo "  uv run robosim view --scene stairs"
echo "  uv run pytest"
