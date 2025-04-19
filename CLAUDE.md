# MCP ChatGPT Server

This MCP server allows you to access OpenAI's ChatGPT API directly from Claude.

## Features

- Call the ChatGPT API with customizable parameters
- Ask Claude and ChatGPT to talk to each other in a long-running discussion
- Configure model versions, temperature, and other parameters
- Use web search to get up-to-date information from the internet
- Uses OpenAI's Responses API for automatic conversation state management
- Requires your own OpenAI API key
- Support for various OpenAI models including O1 family models

## Available Tools

The MCP server provides the following tools:

1. `ask_chatgpt(prompt, model, temperature, max_output_tokens, response_id)` - Send a prompt to ChatGPT and get a response

2. `ask_chatgpt_with_web_search(prompt, model, temperature, max_output_tokens, response_id)` - Send a prompt to ChatGPT with web search enabled to get up-to-date information

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

## Example Usage

### Basic ChatGPT usage:

Ask ChatGPT a question:
```
Use the ask_chatgpt tool to answer: What is the best way to learn Python?
```

Have a conversation between Claude and ChatGPT:
```
Use the ask_chatgpt tool to have a two-way conversation between you and ChatGPT about the topic that is most important to you.
```

### With web search:

For questions that may benefit from up-to-date information:
```
Use the ask_chatgpt_with_web_search tool to answer: What are the latest developments in quantum computing?
```

## Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `DEFAULT_MODEL` (optional): Default model to use (default: "gpt-4o")
- `DEFAULT_TEMPERATURE` (optional): Default temperature setting (default: 0.7)
- `MAX_OUTPUT_TOKENS` (optional): Default maximum output tokens (default: 1000)