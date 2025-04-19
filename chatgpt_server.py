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
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "chatgpt-4o-latest",
    "o1",
    "o1-pro",
    "o3",
    "o3-mini",
    "o4-mini-high"
]

# Initialize FastMCP server
mcp = FastMCP(
    "ChatGPT API",
    dependencies=["openai", "python-dotenv", "httpx", "pydantic"],
)

# Models that don't support temperature
MODELS_WITHOUT_TEMPERATURE = [
    "o1", "o1-pro", 
    "o3", "o3-mini", 
    "o4-mini-high"
]


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
def extract_text_from_response(response, ctx: Optional[Context] = None) -> str:
    """Extract text from various response structures"""
    try:
        # Log response structure for debugging
        response_type = type(response).__name__
        response_attrs = dir(response)
        
        # Log more details if context is provided
        if ctx:
            ctx.info(f"Response type: {response_type}")
            ctx.info(f"Response attributes: {', '.join(attr for attr in response_attrs if not attr.startswith('_'))}")
            if hasattr(response, 'id'):
                ctx.info(f"Response ID: {response.id}")
        
        # If response has output_text attribute, use it directly (most common case)
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
        
        # Try other common attributes
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        
        # Handle case where output might be different structure
        # Return a default message if we can't extract text
        error_msg = "Response received but text content could not be extracted. You can view the response in the API logs."
        if ctx:
            ctx.error(error_msg)
            ctx.info(f"Response structure: {json.dumps(str(response))}")
        return error_msg
    
    except Exception as e:
        error_msg = f"Error extracting text from response: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        else:
            logger.error(error_msg)
        return "Error extracting response text. Please check the logs for details."


# Helper function to create validated API parameters
def create_safe_api_params(model: str, max_output_tokens: int, temperature: Optional[float] = None, additional_params: dict = None) -> dict:
    """Create a dictionary of parameters for OpenAI API with proper validation"""
    # Start with required parameters
    params = {
        "model": model,
        "max_output_tokens": max_output_tokens,
    }
    
    # Only add temperature if the model supports it and a value was provided
    if model not in MODELS_WITHOUT_TEMPERATURE and temperature is not None:
        params["temperature"] = temperature
    
    # Add any additional parameters
    if additional_params:
        # Create a copy to avoid modifying the original
        filtered_params = additional_params.copy()
        
        # If this is a model that doesn't support temperature, ensure it's
        # removed from any nested parameters as well
        if model in MODELS_WITHOUT_TEMPERATURE and "temperature" in filtered_params:
            del filtered_params["temperature"]
            
        params.update(filtered_params)
    
    # Final safety check - ensure temperature is removed for unsupported models
    if model in MODELS_WITHOUT_TEMPERATURE and "temperature" in params:
        del params["temperature"]
    
    return params

# Tools

@mcp.tool()
async def ask_chatgpt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,  # Make temperature optional
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
        # Log key parameters
        ctx.info(f"Model: {model}, Temperature: {temperature if temperature is not None else 'None (using default)'}")
        
        # Create a safe parameter dictionary
        additional_params = {}
        if response_id:
            additional_params["previous_response_id"] = response_id
            additional_params["input"] = [{"role": "user", "content": prompt}]
            ctx.info(f"Continuing conversation with response ID: {response_id}")
        else:
            additional_params["input"] = prompt
            ctx.info(f"Starting new conversation with prompt: {prompt[:50]}...")
            
        # Create safe API parameters
        kwargs = create_safe_api_params(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            additional_params=additional_params
        )
        
        # Log the final API parameters
        ctx.info(f"API parameters: {json.dumps(kwargs)}")
        
        # Call the API
        response = await async_client.responses.create(**kwargs)
        
        # Extract the text content using the helper function
        output_text = extract_text_from_response(response, ctx)
        
        # Return response with ID for reference
        return f"{output_text}\n\n(Response ID: {response.id})"
    
    except Exception as e:
        error_message = f"Error calling ChatGPT API: {str(e)}"
        logger.error(error_message)
        ctx.error(f"API call details: model={model}, params={kwargs}")
        return error_message


@mcp.tool()
async def ask_chatgpt_with_web_search(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,  # Make temperature optional
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
        
        # Log key parameters
        ctx.info(f"Web search with model: {model}, Temperature: {temperature if temperature is not None else 'None (using default)'}")
        
        # Create a safe parameter dictionary
        additional_params = {
            "tools": [web_search_tool]
        }
        
        if response_id:
            additional_params["previous_response_id"] = response_id
            additional_params["input"] = [{"role": "user", "content": prompt}]
            ctx.info(f"Continuing web search conversation with response ID: {response_id}")
        else:
            additional_params["input"] = prompt
            ctx.info(f"Starting new web search conversation with prompt: {prompt[:50]}...")
            
        # Create safe API parameters
        kwargs = create_safe_api_params(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            additional_params=additional_params
        )
        
        # Log the final API parameters
        ctx.info(f"Web search API parameters: {json.dumps(kwargs)}")
        
        # Call the API
        response = await async_client.responses.create(**kwargs)
        
        # Log response for debugging
        logger.info(f"Web search response ID: {response.id}")
        
        # Extract the text content using the helper function
        output_text = extract_text_from_response(response, ctx)
        
        # Return response with ID for reference
        return f"{output_text}\n\n(Response ID: {response.id})"
    
    except Exception as e:
        error_message = f"Error calling ChatGPT with web search: {str(e)}"
        logger.error(error_message)
        ctx.error(f"Web search API call details: model={model}, params={kwargs}")
        return error_message


if __name__ == "__main__":
    print("ChatGPT MCP Server starting...", flush=True)
    logger.info("ChatGPT MCP Server initializing...")
    
    # Log environment variable info (without actual key)
    env_vars = {k: "***" if k == "OPENAI_API_KEY" else v for k, v in os.environ.items() if k.startswith("OPEN") or k.startswith("DEFAULT")}
    logger.info(f"Environment variables: {env_vars}")
    
    # Log model configuration
    logger.info(f"Available models: {AVAILABLE_MODELS}")
    logger.info(f"Models without temperature: {MODELS_WITHOUT_TEMPERATURE}")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Default temperature: {DEFAULT_TEMPERATURE}")
    
    # Run the server
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Fatal error starting MCP server: {str(e)}")
        print(f"Error: {str(e)}", flush=True)