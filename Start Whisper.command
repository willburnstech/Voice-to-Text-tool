#!/bin/bash
# Voice-Paste Launcher
# Navigate to the script's directory (portable)
cd "$(dirname "$0")" || {
    echo "Error: Could not navigate to script directory"
    read -p "Press Enter to exit..."
    exit 1
}

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the app
./venv/bin/python3 whisper.py
