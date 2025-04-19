# ChatGPT MCP Server Tests

This directory contains test scripts for the ChatGPT MCP Server.

## Test Scripts

### simple_test.py

A simple script to test the OpenAI API integration with our server's settings. This script:
- Displays a list of all available models and their temperature parameter support
- Uses the helper function from our main server to create safe API parameters
- Makes a direct call to the OpenAI Responses API
- Displays the response from the API

Run with:
```bash
python simple_test.py [model_name]
```

### test_openai.py

A basic test script for testing direct OpenAI API connectivity with various models. Useful for diagnosing connectivity or authentication issues.

Run with:
```bash
python test_openai.py [model_name]
```

### cli_test.py

A more sophisticated test that simulates a full JSON-RPC request to the MCP server and captures the response. This test:
- Creates a JSON-RPC request object
- Passes it to the server via stdin
- Captures the stdout response
- Parses and displays the result

Run with:
```bash
python cli_test.py
```

## Running All Tests

To run all tests sequentially:

```bash
cd tests
python simple_test.py
python test_openai.py
python cli_test.py
```

## Troubleshooting

If you encounter errors:

1. Ensure the `OPENAI_API_KEY` environment variable is set correctly
2. Verify connectivity to the OpenAI API
3. Check that the model you're testing is available and spelled correctly
4. For models that don't support temperature parameters, verify they're correctly listed in the `MODELS_WITHOUT_TEMPERATURE` array in the main server file