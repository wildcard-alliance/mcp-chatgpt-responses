# MCP ChatGPT Server

This MCP server allows you to access OpenAI's ChatGPT API directly from Claude Desktop.

## Features

- Call the ChatGPT API with customizable parameters
- Pass conversations between Claude and ChatGPT
- Configure model versions, temperature, and other parameters
- Use web search to get up-to-date information from the internet
- Uses OpenAI's Responses API for automatic conversation state management
- Use your own OpenAI API key

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- [Claude Desktop](https://claude.ai/download) application
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) for Python package management

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/billster45/mcp-chatgpt-responses.git
   cd mcp-chatgpt-responses
   ```

2. Install dependencies using uv:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Using with Claude Desktop

1. Configure Claude Desktop to use this MCP server by editing the configuration file at:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following configuration (adjust paths as needed):
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

To ask ChatGPT a question:
```
Use the ask_chatgpt tool to answer: What is the best way to learn Python?
```

To continue a conversation with ChatGPT:
```
Use the ask_chatgpt tool with response_id "resp_abc123" to answer: Can you elaborate more on that point?
```

### With web search:

For questions that may benefit from up-to-date information:
```
Use the ask_chatgpt_with_web_search tool to answer: What are the latest developments in quantum computing?
```

To continue a conversation that uses web search:
```
Use the ask_chatgpt_with_web_search tool with response_id "resp_abc123" to answer: Tell me more about the practical applications of these developments.
```

## How It Works

This tool utilizes OpenAI's Responses API, which automatically maintains conversation state on OpenAI's servers. This approach:

1. Simplifies code by letting OpenAI handle the conversation history
2. Provides more reliable context tracking
3. Improves the user experience by maintaining context across messages
4. Allows access to the latest information from the web with the web search tool

When you start a conversation, you'll receive a response ID (starting with "resp_"). To continue the conversation in your next message, simply include this response ID as a parameter to the `ask_chatgpt` or `ask_chatgpt_with_web_search` tool.

## Configuration

You can customize the server behavior by modifying the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DEFAULT_MODEL`: Default model to use (default: "gpt-4o")
- `DEFAULT_TEMPERATURE`: Default temperature setting (default: 0.7)
- `MAX_OUTPUT_TOKENS`: Maximum tokens in response (default: 1000)

## License

MIT License