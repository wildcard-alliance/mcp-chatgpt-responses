#!/bin/bash
cd "$(dirname "$0")"
# Activate virtual environment from the root directory
source ../../../mcp-venv/bin/activate
python chatgpt_server.py