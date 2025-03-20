#!/usr/bin/env python3
"""
This script helps install the ChatGPT MCP server in Claude Desktop.
It creates or updates the claude_desktop_config.json file with the correct configuration.
"""

import os
import json
import platform
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Install ChatGPT MCP server in Claude Desktop")
    parser.add_argument("--api-key", help="Your OpenAI API key")
    parser.add_argument("--model", default="gpt-4o", help="Default model (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Default temperature (0-2, default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens (default: 1000)")
    args = parser.parse_args()
    
    # Get the path to the Claude Desktop config file
    if platform.system() == "Darwin":  # macOS
        config_path = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
    elif platform.system() == "Windows":
        config_path = os.path.join(os.getenv("APPDATA"), "Claude", "claude_desktop_config.json")
    else:
        print("Unsupported platform. Claude Desktop currently only supports macOS and Windows.")
        return
    
    # Get the absolute path to the chatgpt_server.py file
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "chatgpt_server.py"))
    
    # Create config directory if it doesn't exist
    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)
    
    # Load existing config or create a new one
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            config = {}
    else:
        config = {}
    
    # Initialize mcpServers section if it doesn't exist
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Check for API key
    api_key = args.api_key
    if not api_key:
        # Try to get it from .env file
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.strip().split("=")[1].strip()
                        break
    
    if not api_key:
        print("Warning: No OpenAI API key provided. You will need to add it manually to the config file.")
        api_key = "your_openai_api_key_here"
    
    # Add or update ChatGPT server configuration
    if platform.system() == "Windows":
        # Fix path for Windows
        server_path = server_path.replace("\\", "\\\\")
    
    config["mcpServers"]["chatgpt"] = {
        "command": "python",
        "args": [server_path],
        "env": {
            "OPENAI_API_KEY": api_key,
            "DEFAULT_MODEL": args.model,
            "DEFAULT_TEMPERATURE": str(args.temperature),
            "MAX_TOKENS": str(args.max_tokens)
        }
    }
    
    # Save the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"ChatGPT MCP server configured in Claude Desktop.")
    print(f"Config file: {config_path}")
    print("\nPlease restart Claude Desktop to activate the server.")
    
    if api_key == "your_openai_api_key_here":
        print("\nRemember to update your OpenAI API key in the config file before restarting.")

if __name__ == "__main__":
    main()
