#!/bin/bash
# Navigate to the project directory
cd "$HOME/Desktop/Tools/Whisper" || {
    echo "Error: Could not find project directory at ~/Desktop/Tools/Whisper"
    read -p "Press Enter to exit..."
    exit 1
}

# Run the app
./venv/bin/python3 whisper.py
