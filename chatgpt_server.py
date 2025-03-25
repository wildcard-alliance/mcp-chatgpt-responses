#!/usr/bin/env python3
"""
ChatGPT MCP Server

This MCP server provides tools to interact with OpenAI's ChatGPT API from Claude Desktop.
Uses the OpenAI Responses API for simplified conversation state management.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("chatgpt_server")

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Default settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# Model list for resource
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4.5-preview",
    "o1",
    "o1-pro",
    "o3-mini"
]

# Initialize FastMCP server
mcp = FastMCP(
    "ChatGPT API",
    dependencies=["openai", "python-dotenv", "httpx", "pydantic"],
)

# Models that don't support temperature
MODELS_WITHOUT_TEMPERATURE = ["o1", "o1-pro", "o3-mini"]


class OpenAIRequest(BaseModel):
    """Model for OpenAI API request parameters"""
    model: str = Field(default=DEFAULT_MODEL, description="OpenAI model name")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, description="Temperature (0-2)", ge=0, le=2)
    max_output_tokens: int = Field(default=MAX_OUTPUT_TOKENS, description="Maximum tokens in response", ge=1)
    response_id: Optional[str] = Field(default=None, description="Optional response ID for continuing a chat")


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Initialize and clean up application resources"""
    logger.info("ChatGPT MCP Server starting up")
    try:
        yield {}
    finally:
        logger.info("ChatGPT MCP Server shutting down")


# Resources

@mcp.resource("chatgpt://models")
def available_models() -> str:
    """List available ChatGPT models"""
    return json.dumps(AVAILABLE_MODELS, indent=2)


# Helper function to extract text from response
def extract_text_from_response(response) -> str:
    """Extract text from various response structures"""
    try:
        # Log response structure for debugging
        logger.info(f"Response type: {type(response)}")
        
        # If response has output_text attribute, use it directly
        if hasattr(response, 'output_text'):
            return response.output_text
        
        # If response has output attribute (structured response)
        if hasattr(response, 'output') and response.output:
            # Iterate through output items to find text content
            for output_item in response.output:
                if hasattr(output_item, 'content'):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            return content_item.text
        
        # Handle case where output might be different structure
        # Return a default message if we can't extract text
        return "Response received but text content could not be extracted. You can view the response in the API logs."
    
    except Exception as e:
        logger.error(f"Error extracting text from response: {str(e)}")
        return "Error extracting response text. Please check the logs for details."


# Tools

@mcp.tool()
async def ask_chatgpt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    response_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Send a prompt to ChatGPT and get a response
    
    Args:
        prompt: The message to send to ChatGPT
        model: The OpenAI model to use (default: gpt-4o)
        temperature: Sampling temperature (0-2, default: 0.7)
        max_output_tokens: Maximum tokens in response (default: 1000)
        response_id: Optional response ID for continuing a chat
    
    Returns:
        ChatGPT's response
    """
    ctx.info(f"Calling ChatGPT with model: {model}")
    
    try:
        # Prepare kwargs for the API call
        kwargs = {
            "model": model,
            "max_output_tokens": max_output_tokens,
        }
        
        # Only add temperature if the model supports it
        if model not in MODELS_WITHOUT_TEMPERATURE:
            kwargs["temperature"] = temperature
            ctx.info(f"Using temperature: {temperature}")
        else:
            ctx.info(f"Model {model} does not support temperature parameter, skipping it")
        
        # Format input based on whether this is a new conversation or continuing one
        if response_id:
            # For continuing a conversation
            kwargs["previous_response_id"] = response_id
            kwargs["input"] = [{"role": "user", "content": prompt}]
            response = await async_client.responses.create(**kwargs)
        else:
            # For starting a new conversation
            kwargs["input"] = prompt
            response = await async_client.responses.create(**kwargs)
        
        # Extract the text content using the helper function
        output_text = extract_text_from_response(response)
        
        # Return response with ID for reference
        return f"{output_text}\n\n(Response ID: {response.id})"
    
    except Exception as e:
        error_message = f"Error calling ChatGPT API: {str(e)}"
        logger.error(error_message)
        return error_message


@mcp.tool()
async def ask_chatgpt_with_web_search(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    response_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Send a prompt to ChatGPT with web search capability enabled
    
    Args:
        prompt: The message to send to ChatGPT
        model: The OpenAI model to use (default: gpt-4o)
        temperature: Sampling temperature (0-2, default: 0.7)
        max_output_tokens: Maximum tokens in response (default: 1000)
        response_id: Optional response ID for continuing a chat
    
    Returns:
        ChatGPT's response with information from web search
    """
    ctx.info(f"Calling ChatGPT with web search using model: {model}")
    
    try:
        # Define web search tool
        web_search_tool = {"type": "web_search"}
        
        # Prepare kwargs for the API call
        kwargs = {
            "model": model,
            "max_output_tokens": max_output_tokens,
            "tools": [web_search_tool],
        }
        
        # Only add temperature if the model supports it
        if model not in MODELS_WITHOUT_TEMPERATURE:
            kwargs["temperature"] = temperature
            ctx.info(f"Using temperature: {temperature}")
        else:
            ctx.info(f"Model {model} does not support temperature parameter, skipping it")
        
        # Format input based on whether this is a new conversation or continuing one
        if response_id:
            # For continuing a conversation
            kwargs["previous_response_id"] = response_id
            kwargs["input"] = [{"role": "user", "content": prompt}]
            response = await async_client.responses.create(**kwargs)
        else:
            # For starting a new conversation
            kwargs["input"] = prompt
            response = await async_client.responses.create(**kwargs)
        
        # Log response for debugging
        logger.info(f"Web search response ID: {response.id}")
        logger.info(f"Web search response structure: {dir(response)}")
        
        # Extract the text content using the helper function
        output_text = extract_text_from_response(response)
        
        # Return response with ID for reference
        return f"{output_text}\n\n(Response ID: {response.id})"
    
    except Exception as e:
        error_message = f"Error calling ChatGPT with web search: {str(e)}"
        logger.error(error_message)
        return error_message


if __name__ == "__main__":
    # Run the server
    mcp.run(transport='stdio')