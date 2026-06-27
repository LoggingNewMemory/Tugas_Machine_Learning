#!/bin/bash

echo "================================================="
echo "🤖 Auto-Setup for Qwen Local Server "
echo "================================================="

VENV_DIR="qwen_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Checking dependencies..."
if ! python3 -c "import vllm" &> /dev/null; then
    echo "Installing required packages (vllm, pyngrok)..."
    python3 -m pip install --upgrade pip >/dev/null
    pip install vllm pyngrok
else
    echo "Dependencies already installed."
fi

echo "Starting Server..."
python3 start_qwen_server.py
