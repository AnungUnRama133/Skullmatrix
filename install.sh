#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv"
BIN_DIR="$HOME/.local/bin"

echo "=== Skull Renderer Installer ==="

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed."
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv "$VENV"

echo "[2/4] Installing Python dependencies..."
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "[3/4] Installing skull command..."
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/skull" <<EOF2
#!/bin/bash
exec "$VENV/bin/python" "$PROJECT_DIR/skullface.py" "\$@"
EOF2

chmod +x "$BIN_DIR/skull"

echo "[4/4] Done!"
echo
echo "Run the skull with:"
echo
echo "    $BIN_DIR/skull"
echo
echo "If ~/.local/bin is already in your PATH, simply type:"
echo
echo "    skull"
