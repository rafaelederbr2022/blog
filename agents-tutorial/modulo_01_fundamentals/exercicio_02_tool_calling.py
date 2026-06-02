"""
Module 1 - Exercise 02: Tool Calling (Function Calling)
========================================================
Learn how LLMs can decide which tool to invoke based on user queries,
and how to bind custom tools to a model using LangChain's tool system.

Concepts covered:
- Defining tools using the @tool decorator from langchain_core.tools
- Binding tools to an LLM with .bind_tools()
- How the LLM decides which tool to call (Function Calling)
- Parsing tool_calls from AIMessage
- Executing the selected tool and displaying results
"""

import sys
sys.path.append('..')
from config import load_config, get_llm

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city. Returns a weather report string."""
    # Mock weather data for demonstration purposes
    weather_data = {
        "new york": "Sunny, 22°C, humidity 45%",
        "london": "Cloudy, 15°C, humidity 78%",
        "tokyo": "Rainy, 18°C, humidity 90%",
        "paris": "Partly cloudy, 20°C, humidity 55%",
        "são paulo": "Warm, 28°C, humidity 60%",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather in {city}: Clear skies, 25°C, humidity 50% (default data)"


# ---------------------------------------------------------------------------
# Part 1: Defining and inspecting tools
# ---------------------------------------------------------------------------

def demonstrate_tool_definitions():
    """Show how tools are defined and what metadata they carry."""
    print("\n" + "=" * 60)
    print("PART 1: Defining Tools with @tool Decorator")
    print("=" * 60)
    print()
    print("The @tool decorator from langchain_core.tools converts a regular")
    print("Python function into a LangChain Tool. The function's docstring")
    print("becomes the tool description that the LLM uses to decide when")
    print("to call it.")
    print()

    tools = [add, multiply, get_weather]

    for t in tools:
        print(f"Tool: {t.name}")
        print(f"  Description: {t.description}")
        print(f"  Args schema: {t.args}")
        print()

    print("The LLM reads these descriptions to understand what each tool")
    print("does and what arguments it expects.")


# ---------------------------------------------------------------------------
# Part 2: Binding tools to the LLM
# ---------------------------------------------------------------------------

def demonstrate_bind_tools(llm):
    """Show how to bind tools to an LLM so it can choose which to call."""
    print("\n" + "=" * 60)
    print("PART 2: Binding Tools to the LLM")
    print("=" * 60)
    print()
    print("Use .bind_tools() to attach tools to an LLM. This tells the")
    print("model which tools are available. The model can then decide")
    print("whether to call a tool (and which one) based on the user query.")
    print()

    tools = [add, multiply, get_weather]
    llm_with_tools = llm.bind_tools(tools)

    print(f"Bound {len(tools)} tools to the LLM: {[t.name for t in tools]}")
    print()
    print("Now the LLM knows about these tools and can request to call")
    print("them when appropriate.")

    return llm_with_tools


# ---------------------------------------------------------------------------
# Part 3: LLM choosing which tool to call
# ---------------------------------------------------------------------------

def demonstrate_tool_selection(llm_with_tools):
    """Show how the LLM selects the appropriate tool based on the query."""
    print("\n" + "=" * 60)
    print("PART 3: LLM Choosing Which Tool to Call")
    print("=" * 60)
    print()
    print("When you send a message to an LLM with bound tools, it analyzes")
    print("the query and decides if a tool call is needed. If so, it returns")
    print("an AIMessage with tool_calls populated instead of plain text.")
    print()

    queries = [
        "What is 15 plus 28?",
        "Calculate 7 multiplied by 9",
        "What's the weather like in Tokyo?",
    ]

    results = []

    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 40)

        response = llm_with_tools.invoke([HumanMessage(content=query)])

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  Tool selected: {tc['name']}")
                print(f"  Arguments: {tc['args']}")
                results.append((query, response))
        else:
            print(f"  No tool call - LLM responded with text:")
            print(f"  {response.content[:100]}")
            results.append((query, response))

        print()

    return results


# ---------------------------------------------------------------------------
# Part 4: Extracting and executing tool calls
# ---------------------------------------------------------------------------

def demonstrate_tool_execution(llm_with_tools):
    """Show how to extract tool_calls from AIMessage and execute them."""
    print("\n" + "=" * 60)
    print("PART 4: Extracting and Executing Tool Calls")
    print("=" * 60)
    print()
    print("After the LLM returns tool_calls, you need to:")
    print("  1. Extract the tool name and arguments from the response")
    print("  2. Find the matching tool function")
    print("  3. Execute it with the provided arguments")
    print("  4. Return the result to the user (or back to the LLM)")
    print()

    # Map tool names to tool functions for lookup
    tools = [add, multiply, get_weather]
    tool_map = {t.name: t for t in tools}

    # Test queries that will trigger different tools
    test_cases = [
        "What is 123 plus 456?",
        "Multiply 12 by 15",
        "What's the weather in London right now?",
    ]

    for query in test_cases:
        print(f"User query: '{query}'")
        print("-" * 40)

        # Step 1: Send query to LLM with tools
        response = llm_with_tools.invoke([HumanMessage(content=query)])

        # Step 2: Check if the LLM wants to call a tool
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                print(f"  LLM requested tool: '{tool_name}'")
                print(f"  With arguments: {tool_args}")

                # Step 3: Execute the tool
                selected_tool = tool_map[tool_name]
                result = selected_tool.invoke(tool_args)

                # Step 4: Display the result
                print(f"  Tool result: {result}")
        else:
            print(f"  LLM responded directly: {response.content[:100]}")

        print()


# ---------------------------------------------------------------------------
# Part 5: Complete tool calling flow
# ---------------------------------------------------------------------------

def demonstrate_complete_flow(llm_with_tools):
    """Demonstrate a complete tool calling flow with multiple interactions."""
    print("\n" + "=" * 60)
    print("PART 5: Complete Tool Calling Flow")
    print("=" * 60)
    print()
    print("This demonstrates the full cycle: user query -> LLM decides ->")
    print("tool executes -> result returned. This is the foundation for")
    print("building AI agents that can take actions in the real world.")
    print()

    tools = [add, multiply, get_weather]
    tool_map = {t.name: t for t in tools}

    # A query that doesn't need a tool
    print("Example 1: Query that doesn't need a tool")
    print("-" * 40)
    query = "What is the capital of France?"
    print(f"Query: '{query}'")
    response = llm_with_tools.invoke([HumanMessage(content=query)])

    if response.tool_calls:
        print(f"  Tool called: {response.tool_calls[0]['name']}")
    else:
        print(f"  LLM answered directly (no tool needed):")
        print(f"  {response.content[:150]}")
    print()

    # A query that needs a tool
    print("Example 2: Query that requires a tool")
    print("-" * 40)
    query = "I need to add 999 and 1001, what's the result?"
    print(f"Query: '{query}'")
    response = llm_with_tools.invoke([HumanMessage(content=query)])

    if response.tool_calls:
        tc = response.tool_calls[0]
        print(f"  LLM chose tool: '{tc['name']}'")
        print(f"  Arguments: {tc['args']}")
        result = tool_map[tc["name"]].invoke(tc["args"])
        print(f"  Execution result: {result}")
    else:
        print(f"  LLM answered directly: {response.content[:150]}")
    print()

    # Show the raw tool_calls structure
    print("Example 3: Inspecting the raw tool_calls structure")
    print("-" * 40)
    query = "What's the weather in Paris?"
    print(f"Query: '{query}'")
    response = llm_with_tools.invoke([HumanMessage(content=query)])

    print(f"  response.tool_calls type: {type(response.tool_calls)}")
    if response.tool_calls:
        print(f"  Number of tool calls: {len(response.tool_calls)}")
        tc = response.tool_calls[0]
        print(f"  Tool call structure:")
        print(f"    'name': '{tc['name']}'")
        print(f"    'args': {tc['args']}")
        print(f"    'id': '{tc['id']}'")
        result = tool_map[tc["name"]].invoke(tc["args"])
        print(f"  Executed result: {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all tool calling demonstrations."""
    print("=" * 60)
    print("  LangChain Fundamentals: Tool Calling (Function Calling)")
    print("=" * 60)

    # Load configuration and initialize LLM
    config = load_config()
    llm = get_llm()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Temperature: {config['default_temperature']}")

    # Part 1: Show tool definitions
    demonstrate_tool_definitions()

    # Part 2: Bind tools to LLM
    llm_with_tools = demonstrate_bind_tools(llm)

    # Part 3: LLM selects tools based on queries
    demonstrate_tool_selection(llm_with_tools)

    # Part 4: Execute tool calls
    demonstrate_tool_execution(llm_with_tools)

    # Part 5: Complete flow
    demonstrate_complete_flow(llm_with_tools)

    print("\n" + "=" * 60)
    print("  Exercise 02 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. @tool decorator: Converts functions into LangChain tools")
    print("  2. .bind_tools(): Attaches tools to an LLM for function calling")
    print("  3. Tool selection: LLM reads tool descriptions to pick the right one")
    print("  4. tool_calls: AIMessage contains structured tool call requests")
    print("  5. Execution: You extract args, run the tool, and return results")
    print()


if __name__ == "__main__":
    main()
