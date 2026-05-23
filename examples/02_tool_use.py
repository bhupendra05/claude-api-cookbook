"""Tool use with proper tool_result blocks, multi-turn tool execution loop."""
import anthropic
import json

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information. Returns a summary of results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-5)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '2 + 2', '100 * 1.08'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor (mock implementations)
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result as a string."""
    if tool_name == "search_web":
        query = tool_input["query"]
        num = tool_input.get("num_results", 3)
        return (
            f"Search results for '{query}' ({num} results):\n"
            f"1. Python is a high-level programming language created by Guido van Rossum.\n"
            f"2. Python 3.12 introduced significant performance improvements.\n"
            f"3. Python is widely used in AI/ML, web development, and data science."
        )

    elif tool_name == "calculate":
        expression = tool_input["expression"]
        try:
            # Safe eval with limited builtins
            allowed = {k: getattr(__builtins__, k, None) for k in ["abs", "round", "min", "max", "sum", "pow"]}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"

    elif tool_name == "get_weather":
        city = tool_input["city"]
        units = tool_input.get("units", "celsius")
        temp = 22 if units == "celsius" else 72
        unit_sym = "°C" if units == "celsius" else "°F"
        return f"Weather in {city}: {temp}{unit_sym}, partly cloudy, humidity 65%, wind 12 km/h"

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Tool use loop
# ---------------------------------------------------------------------------

def run_with_tools(user_message: str, model: str = "claude-opus-4-5") -> str:
    """Run a conversation with tool use, handling multi-turn tool calls."""
    messages = [{"role": "user", "content": user_message}]

    print(f"User: {user_message}")
    print("-" * 60)

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        print(f"Stop reason: {response.stop_reason}")

        # Add assistant's response to conversation
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        elif response.stop_reason == "tool_use":
            # Execute all requested tools
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool: {block.name}({json.dumps(block.input, indent=2)})")
                    result = execute_tool(block.name, block.input)
                    print(f"  Result: {result[:100]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

        else:
            break

    return "Conversation ended unexpectedly."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "What is Python? Also, what is 15% of 299.99?",
        "What's the weather in Tokyo and Paris right now? And search for the best Python web frameworks.",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        result = run_with_tools(query)
        print(f"\nFinal answer:\n{result}")
