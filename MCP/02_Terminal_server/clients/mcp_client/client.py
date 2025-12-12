# Import necessary libraries
import asyncio  # For handling asynchronous operations
import os       # For environment variable access
import sys      # For system-specific parameters and functions
import json     # For handling JSON data (used when printing function declarations)

# Import MCP client components
from typing import Optional  # For type hinting optional values
from contextlib import AsyncExitStack  # For managing multiple async tasks
from mcp import ClientSession, StdioServerParameters  # MCP session management
from mcp.client.stdio import stdio_client  # MCP client for standard I/O communication

from dotenv import load_dotenv  # For loading API keys from a .env file

# Groq client (OpenAI-compatible)
from groq import Groq

# Load environment variables from .env file
load_dotenv()


class MCPClient:
    def __init__(self):
        """Initialize the MCP client and configure the Groq API."""
        self.session: Optional[ClientSession] = None  # MCP session for communication
        self.exit_stack = AsyncExitStack()  # Manages async resource cleanup

        # Retrieve the Groq API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Please add GROQ_API_KEY to your .env file.")

        # Configure the Groq AI client
        self.groq_client = Groq(api_key=groq_api_key)

        # Allow overriding the model; default to the current Groq recommended model
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        # This will hold OpenAI-style tool definitions
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and list available tools."""

        # Determine whether the server script is written in Python or JavaScript
        command = "python" if server_script_path.endswith('.py') else "node"

        # Define the parameters for connecting to the MCP server
        server_params = StdioServerParameters(command=command, args=[server_script_path])

        # Establish communication with the MCP server using standard input/output (stdio)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        # Extract the read/write streams from the transport object
        self.stdio, self.write = stdio_transport

        # Initialize the MCP client session, which allows interaction with the server
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # Send an initialization request to the MCP server
        await self.session.initialize()

        # Request the list of available tools from the MCP server
        response = await self.session.list_tools()
        tools = response.tools  # Extract the tool list from the response

        # Print a message showing the names of the tools available on the server
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Convert MCP tools to OpenAI/Groq-style tools
        self.tools = convert_mcp_tools_to_groq(tools)

    async def process_query(self, query: str) -> str:
        """
        Process a user query using Groq and execute MCP tool calls if needed.
        """

        # Basic OpenAI-style messages format
        messages = [
            {"role": "user", "content": query}
        ]

        # First call: let the model decide whether to call tools
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            tools=self.tools,
        )

        msg = response.choices[0].message

        # If there are no tool calls, just return the text
        if not getattr(msg, "tool_calls", None):
            return msg.content or ""

        # Otherwise, handle tool calls
        tool_call_results = []

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            # arguments is a JSON string in OpenAI-style tools
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"\n[Groq requested tool call: {tool_name} with args {tool_args}]")

            try:
                # Adapt common *nix commands to Windows if needed for run_command
                if tool_name == "run_command" and isinstance(tool_args, dict) and "command" in tool_args:
                    tool_args["command"] = _adapt_command_for_windows(tool_args["command"])

                result = await self.session.call_tool(tool_name, tool_args)
                serialized = _serialize_tool_content(result.content)
                function_response = {"result": serialized}
            except Exception as e:
                function_response = {"error": str(e)}

            # This is the tool result message we send back to the model
            tool_call_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(function_response),
                }
            )

        # Second call: send user message, model's tool_call message, and tool results
        messages.append(
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        messages.extend(tool_call_results)

        final_response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            tools=self.tools,
        )

        final_msg = final_response.choices[0].message
        return final_msg.content or ""

    async def chat_loop(self):
        """Run an interactive chat session with the user."""
        print("\nMCP Client Started! Type 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            # Process the user's query and display the response
            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up resources before exiting."""
        await self.exit_stack.aclose()


def clean_schema(schema):
    """
    Recursively removes 'title' fields from the JSON schema.

    Args:
        schema (dict): The schema dictionary.

    Returns:
        dict: Cleaned schema without 'title' fields.
    """
    if isinstance(schema, dict):
        schema.pop("title", None)  # Remove title if present

        # Recursively clean nested properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])

    return schema


def convert_mcp_tools_to_groq(mcp_tools):
    """
    Converts MCP tool definitions to OpenAI/Groq-style tools.

    Args:
        mcp_tools (list): List of MCP tool objects with 'name', 'description', and 'inputSchema'.

    Returns:
        list: List of OpenAI-style tool definitions.
    """
    groq_tools = []

    for tool in mcp_tools:
        parameters = clean_schema(tool.inputSchema)

        groq_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": parameters,
                },
            }
        )

    return groq_tools


def _serialize_tool_content(content):
    """
    Convert MCP tool content into something JSON-serializable (primarily text).
    Handles lists of TextContent objects and primitives.
    """
    if content is None:
        return ""

    # If we received a list of content parts, flatten to text lines.
    if isinstance(content, list):
        parts = [_serialize_tool_content(item) for item in content]
        # Filter out empty strings to avoid redundant newlines
        parts = [p for p in parts if p not in (None, "")]
        # If everything was primitive/serializable, keep list; otherwise join to string
        if all(isinstance(p, (str, int, float, bool, dict)) for p in parts):
            # If dicts are present, fall back to string to avoid mixed types confusion
            if any(isinstance(p, dict) for p in parts):
                return "\n".join(str(p) for p in parts)
            return "\n".join(str(p) for p in parts)
        return "\n".join(str(p) for p in parts)

    # TextContent-like objects: prefer .text if present
    if hasattr(content, "text"):
        return getattr(content, "text")

    # If already JSON-serializable primitive, return as-is
    if isinstance(content, (str, int, float, bool, dict)):
        return content

    # Fallback: best-effort string conversion
    return str(content)


def _adapt_command_for_windows(command: str) -> str:
    """
    Best-effort adaptation of common *nix commands to Windows-friendly versions.
    Currently replaces `touch <file>` with `type nul > <file>` to work in
    PowerShell / cmd.
    """
    if os.name != "nt" or not isinstance(command, str):
        return command

    # Replace any occurrence of "touch <file>" with an equivalent create/overwrite
    tokens = command.split("&&")
    new_tokens = []
    for tok in tokens:
        stripped = tok.strip()
        if stripped.startswith("touch "):
            file_part = stripped[len("touch "):].strip().replace("/", "\\")
            new_tokens.append(f"type nul > {file_part}")
        else:
            new_tokens.append(stripped.replace("/", "\\"))

    adapted = " & ".join(new_tokens)  # cmd.exe prefers single ampersand for chaining
    return f'cmd /c "{adapted}"'


async def main():
    """Main function to start the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        # Connect to the MCP server and start the chat loop
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        # Ensure resources are cleaned up
        await client.cleanup()


if __name__ == "__main__":
    # Run the main function within the asyncio event loop
    asyncio.run(main())
