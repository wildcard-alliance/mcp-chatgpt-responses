#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status
set -x  # Print commands and their arguments as they are executed

cd "$(dirname "$0")"
echo "Current directory: $(pwd)"

# Check if virtual environment exists
if [ ! -f "../../../mcp-venv/bin/activate" ]; then
    echo "Virtual environment not found, creating one..."
    cd ../../..
    python -m venv mcp-venv
    cd - > /dev/null
fi

# Activate virtual environment from the root directory
echo "Activating virtual environment..."
source ../../../mcp-venv/bin/activate

# Install or update dependencies
echo "Installing/updating dependencies..."
pip install -r requirements.txt

# Run the server with error output
echo "Starting ChatGPT server..."
python chatgpt_server.py 2>&1