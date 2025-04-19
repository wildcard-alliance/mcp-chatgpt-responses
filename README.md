# MCP ChatGPT Server

This MCP server allows you to access OpenAI's ChatGPT API directly from Claude Desktop.

📝 **Read about why I built this project**: [I Built an AI That Talks to Other AIs: Demystifying the MCP Hype](https://medium.com/@billcockerill/i-built-an-ai-that-talks-to-other-ais-demystifying-the-mcp-hype-88dc03520552)

## Supported Models

- `gpt-4o` (default)
- `gpt-4.5-preview`
- `gpt-4.1`
- `gpt-4.1-2025-04-14`
- `chatgpt-4o-latest`
- `o1`
- `o1-pro`
- `o3`
- `o3-mini`
- `o4-mini-high`

Note: Several models including O1 family models (o1, o1-pro), O3 family models (o3, o3-mini), and O4 family models (o4-mini-high) do not support the temperature parameter, which is automatically omitted when using these models.

## Features

- Call the ChatGPT API with customisable parameters
- Aks Claude and ChatGPT to talk to each other in a long running discussion!
- Configure model versions, temperature, and other parameters
- Use web search to get up-to-date information from the internet
- Uses OpenAI's Responses API for automatic conversation state management
- Use your own OpenAI API key

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- [Claude Desktop](https://claude.ai/download) application
- [OpenAI API key](https://platform.openai.com/settings/organization/api-keys)
- [uv](https://github.com/astral-sh/uv) for Python package management

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/billster45/mcp-chatgpt-responses.git
   cd mcp-chatgpt-responses
   ```

2. Set up a virtual environment and install dependencies using uv:
   ```bash
   uv venv
   ```

   ```bash
   .venv\\Scripts\\activate
   ```
   
   ```bash
   uv pip install -r requirements.txt
   ```

### Using with Claude Desktop

1. Configure Claude Desktop to use this MCP server by following the instructions at:
   [MCP Quickstart Guide](https://modelcontextprotocol.io/quickstart/user#2-add-the-filesystem-mcp-server)

2. Add the following configuration to your Claude Desktop config file (adjust paths as needed):
   ```json
   {
     "mcpServers": {
       "chatgpt": {
         "command": "uv",
         "args": [
           "--directory",
           "\\path\\to\\mcp-chatgpt-responses",
           "run",
           "chatgpt_server.py"
         ],
         "env": {
           "OPENAI_API_KEY": "your-api-key-here",
           "DEFAULT_MODEL": "gpt-4o",
           "DEFAULT_TEMPERATURE": "0.7",
           "MAX_TOKENS": "1000"
         }
       }
     }
   }
   ```

3. Restart Claude Desktop.

4. You can now use the ChatGPT API through Claude by asking questions that mention ChatGPT or that Claude might not be able to answer.

## Available Tools

The MCP server provides the following tools:

1. `ask_chatgpt(prompt, model, temperature, max_output_tokens, response_id)` - Send a prompt to ChatGPT and get a response

2. `ask_chatgpt_with_web_search(prompt, model, temperature, max_output_tokens, response_id)` - Send a prompt to ChatGPT with web search enabled to get up-to-date information

## Example Usage

### Basic ChatGPT usage:

Tell Claude to ask ChatGPT a question!
```
Use the ask_chatgpt tool to answer: What is the best way to learn Python?
```

Tell Claude to have a conversation with ChatGPT:
```
Use the ask_chatgpt tool to have a two way conversation between you and ChatGPT about the topic that is most important to you.
```
Note how in a turn taking conversation the response id allows ChatGPT to store the history of the conversation so its a genuine conversation and not just as series of API calls. This is called [conversation state](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses#openai-apis-for-conversation-state).

### With web search:

For questions that may benefit from up-to-date information:
```
Use the ask_chatgpt_with_web_search tool to answer: What are the latest developments in quantum computing?
```

Now try web search in agentic way to plan your perfect day out based on the weather!
```
Use the ask_chatgpt_with_web_search tool to find the weather tomorrow in New York, then based on that weather and what it returns, keep using the tool to build up a great day out for someone who loves food and parks
```

## How It Works

This tool utilizes OpenAI's Responses API, which automatically maintains conversation state on OpenAI's servers. This approach:

1. Simplifies code by letting OpenAI handle the conversation history
2. Provides more reliable context tracking
3. Improves the user experience by maintaining context across messages
4. Allows access to the latest information from the web with the web search tool

## Project Structure

- `/` - Root directory containing the main server implementation
  - `chatgpt_server.py` - The main MCP server implementation
  - `chatgpt-wrapper.sh` - Shell script for starting the server
  - `requirements.txt` - Python dependencies
- `/tests` - Test scripts and utilities
  - `simple_test.py` - Simple test script for the OpenAI API
  - `test_openai.py` - Direct test for OpenAI API connectivity
  - `cli_test.py` - CLI test for the full MCP server

## Testing

See the [tests/README.md](tests/README.md) file for detailed information on running and using the test scripts.

## License

MIT License