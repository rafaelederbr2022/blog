"""
Module 3 - Exercise 15: StateGraph Basics
==========================================
Learn how to create and execute a simple StateGraph using LangGraph.
A StateGraph is the core abstraction in LangGraph that allows you to define
workflows as directed graphs where nodes are functions and edges define
the execution flow.

Concepts covered:
- Creating a StateGraph with TypedDict state
- Adding nodes (functions that modify state)
- Adding edges connecting nodes
- Setting entry point (START) and finish point (END)
- Compiling and invoking the graph
- Understanding the graph execution flow step by step

Example: A text processing pipeline (input -> process -> format -> output)
"""

import sys
sys.path.append('..')
from config import load_config

from typing import TypedDict
from langgraph.graph import StateGraph, END, START


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class TextProcessingState(TypedDict):
    """State for our text processing pipeline.

    Each field represents a piece of data that flows through the graph.
    Nodes can read and write to these fields.
    """
    raw_input: str
    processed_text: str
    formatted_output: str
    steps_log: list[str]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def input_node(state: TextProcessingState) -> dict:
    """First node: receives raw input and prepares it for processing.

    Nodes receive the current state and return a dict with fields to update.
    Only the returned fields are updated; other fields remain unchanged.
    """
    raw = state["raw_input"]
    log_entry = f"[input_node] Received raw input: '{raw}'"
    print(f"  -> {log_entry}")

    return {
        "processed_text": raw.strip(),
        "steps_log": state.get("steps_log", []) + [log_entry],
    }


def process_node(state: TextProcessingState) -> dict:
    """Second node: processes the text (normalize whitespace, lowercase).

    This simulates a processing step that transforms the data.
    """
    text = state["processed_text"]

    # Normalize: collapse multiple spaces, convert to lowercase
    normalized = " ".join(text.split()).lower()

    log_entry = f"[process_node] Normalized text: '{normalized}'"
    print(f"  -> {log_entry}")

    return {
        "processed_text": normalized,
        "steps_log": state["steps_log"] + [log_entry],
    }


def format_node(state: TextProcessingState) -> dict:
    """Third node: formats the processed text for final output.

    Applies title case and adds decorative formatting.
    """
    text = state["processed_text"]

    # Format: title case and add decoration
    formatted = f"*** {text.title()} ***"

    log_entry = f"[format_node] Formatted output: '{formatted}'"
    print(f"  -> {log_entry}")

    return {
        "formatted_output": formatted,
        "steps_log": state["steps_log"] + [log_entry],
    }


# ---------------------------------------------------------------------------
# Part 1: Building and Running a Basic StateGraph
# ---------------------------------------------------------------------------

def demonstrate_basic_stategraph():
    """Show how to create, compile, and invoke a basic StateGraph."""
    print("\n" + "=" * 60)
    print("PART 1: Building a Basic StateGraph")
    print("=" * 60)
    print()
    print("A StateGraph consists of:")
    print("  1. State: A TypedDict defining the data that flows through the graph")
    print("  2. Nodes: Functions that receive state and return updates")
    print("  3. Edges: Connections that define execution order")
    print("  4. Entry/Exit: START and END markers for the flow")
    print()

    # Step 1: Create the StateGraph with our state schema
    print("Step 1: Create StateGraph with TextProcessingState schema")
    print("-" * 40)
    graph_builder = StateGraph(TextProcessingState)
    print("  graph_builder = StateGraph(TextProcessingState)")
    print()

    # Step 2: Add nodes to the graph
    print("Step 2: Add nodes (each node is a function)")
    print("-" * 40)
    graph_builder.add_node("input", input_node)
    graph_builder.add_node("process", process_node)
    graph_builder.add_node("format", format_node)
    print("  graph_builder.add_node('input', input_node)")
    print("  graph_builder.add_node('process', process_node)")
    print("  graph_builder.add_node('format', format_node)")
    print()

    # Step 3: Add edges connecting nodes
    print("Step 3: Add edges (define execution order)")
    print("-" * 40)
    graph_builder.add_edge(START, "input")
    graph_builder.add_edge("input", "process")
    graph_builder.add_edge("process", "format")
    graph_builder.add_edge("format", END)
    print("  START -> 'input' -> 'process' -> 'format' -> END")
    print()
    print("  add_edge(START, 'input')   # Entry point")
    print("  add_edge('input', 'process')")
    print("  add_edge('process', 'format')")
    print("  add_edge('format', END)    # Exit point")
    print()

    # Step 4: Compile the graph
    print("Step 4: Compile the graph into a runnable")
    print("-" * 40)
    graph = graph_builder.compile()
    print("  graph = graph_builder.compile()")
    print("  The compiled graph is a Runnable that can be invoked.")
    print()

    # Step 5: Invoke the graph with initial state
    print("Step 5: Invoke the graph with initial state")
    print("-" * 40)
    print()

    initial_state = {
        "raw_input": "  Hello   WORLD   from   LangGraph!  ",
        "processed_text": "",
        "formatted_output": "",
        "steps_log": [],
    }

    print(f"  Initial state:")
    print(f"    raw_input: '{initial_state['raw_input']}'")
    print(f"    processed_text: '{initial_state['processed_text']}'")
    print(f"    formatted_output: '{initial_state['formatted_output']}'")
    print()
    print("  Executing graph (watch the nodes fire in order):")
    print()

    # Invoke the graph
    final_state = graph.invoke(initial_state)

    print()
    print("  Final state after graph execution:")
    print(f"    raw_input: '{final_state['raw_input']}'")
    print(f"    processed_text: '{final_state['processed_text']}'")
    print(f"    formatted_output: '{final_state['formatted_output']}'")
    print()
    print("  Execution log:")
    for step in final_state["steps_log"]:
        print(f"    - {step}")

    return graph


# ---------------------------------------------------------------------------
# Part 2: Understanding Graph Execution Flow
# ---------------------------------------------------------------------------

def demonstrate_execution_flow():
    """Show the step-by-step execution flow of a StateGraph."""
    print("\n" + "=" * 60)
    print("PART 2: Understanding Graph Execution Flow")
    print("=" * 60)
    print()
    print("When you invoke a StateGraph, the following happens:")
    print()
    print("  1. The initial state is passed to the entry node")
    print("  2. Each node receives the FULL current state")
    print("  3. Each node returns a PARTIAL dict with fields to update")
    print("  4. The state is updated (merged) with the node's return value")
    print("  5. The updated state is passed to the next node via edges")
    print("  6. When END is reached, the final state is returned")
    print()
    print("Key insight: Nodes do NOT replace the entire state.")
    print("They only update the fields they return.")
    print()

    # Create a simple counter graph to demonstrate state updates
    class CounterState(TypedDict):
        value: int
        history: list[str]

    def add_one(state: CounterState) -> dict:
        new_val = state["value"] + 1
        print(f"  -> add_one: {state['value']} -> {new_val}")
        return {
            "value": new_val,
            "history": state["history"] + [f"add_one({state['value']}) = {new_val}"],
        }

    def multiply_two(state: CounterState) -> dict:
        new_val = state["value"] * 2
        print(f"  -> multiply_two: {state['value']} -> {new_val}")
        return {
            "value": new_val,
            "history": state["history"] + [f"multiply_two({state['value']}) = {new_val}"],
        }

    def add_ten(state: CounterState) -> dict:
        new_val = state["value"] + 10
        print(f"  -> add_ten: {state['value']} -> {new_val}")
        return {
            "value": new_val,
            "history": state["history"] + [f"add_ten({state['value']}) = {new_val}"],
        }

    # Build the graph: add_one -> multiply_two -> add_ten
    print("Building a counter graph: add_one -> multiply_two -> add_ten")
    print("-" * 40)
    print()

    builder = StateGraph(CounterState)
    builder.add_node("add_one", add_one)
    builder.add_node("multiply_two", multiply_two)
    builder.add_node("add_ten", add_ten)

    builder.add_edge(START, "add_one")
    builder.add_edge("add_one", "multiply_two")
    builder.add_edge("multiply_two", "add_ten")
    builder.add_edge("add_ten", END)

    graph = builder.compile()

    # Run with initial value = 5
    # Expected: 5 -> 6 -> 12 -> 22
    print("Running with initial value = 5:")
    print("  Expected: 5 -> add_one -> 6 -> multiply_two -> 12 -> add_ten -> 22")
    print()

    result = graph.invoke({"value": 5, "history": []})

    print()
    print(f"  Final value: {result['value']}")
    print(f"  History: {result['history']}")
    print()

    # Run with a different initial value
    print("Running with initial value = 3:")
    print("  Expected: 3 -> add_one -> 4 -> multiply_two -> 8 -> add_ten -> 18")
    print()

    result2 = graph.invoke({"value": 3, "history": []})

    print()
    print(f"  Final value: {result2['value']}")
    print(f"  History: {result2['history']}")


# ---------------------------------------------------------------------------
# Part 3: Graph with Multiple Fields
# ---------------------------------------------------------------------------

def demonstrate_multi_field_graph():
    """Show a graph where different nodes modify different state fields."""
    print("\n" + "=" * 60)
    print("PART 3: Graph with Multiple State Fields")
    print("=" * 60)
    print()
    print("Each node can read any field but only needs to return the fields")
    print("it wants to update. Unreturned fields remain unchanged.")
    print()

    class DocumentState(TypedDict):
        title: str
        content: str
        word_count: int
        summary: str

    def analyze_content(state: DocumentState) -> dict:
        """Count words in the content."""
        words = len(state["content"].split())
        print(f"  -> analyze_content: counted {words} words")
        return {"word_count": words}

    def generate_summary(state: DocumentState) -> dict:
        """Generate a simple summary (first sentence + word count)."""
        first_sentence = state["content"].split(".")[0] + "."
        summary = f"{first_sentence} (Total: {state['word_count']} words)"
        print(f"  -> generate_summary: '{summary}'")
        return {"summary": summary}

    # Build graph
    builder = StateGraph(DocumentState)
    builder.add_node("analyze", analyze_content)
    builder.add_node("summarize", generate_summary)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "summarize")
    builder.add_edge("summarize", END)

    graph = builder.compile()

    # Invoke
    initial = {
        "title": "LangGraph Introduction",
        "content": "LangGraph is a library for building stateful applications. It uses directed graphs to model workflows. Each node processes state and passes it forward.",
        "word_count": 0,
        "summary": "",
    }

    print(f"  Input document: '{initial['title']}'")
    print(f"  Content: '{initial['content'][:50]}...'")
    print()

    result = graph.invoke(initial)

    print()
    print("  Final state:")
    print(f"    title: '{result['title']}' (unchanged - no node modified it)")
    print(f"    word_count: {result['word_count']} (set by analyze node)")
    print(f"    summary: '{result['summary']}' (set by summarize node)")
    print()
    print("  Notice: 'title' was never returned by any node, so it stayed the same.")
    print("  Each node only updates what it returns.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all StateGraph basic demonstrations."""
    print("=" * 60)
    print("  LangGraph: StateGraph Basics")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates StateGraph fundamentals.")
    print("No LLM calls are needed - we focus on graph structure and flow.")

    # Part 1: Building and running a basic StateGraph
    demonstrate_basic_stategraph()

    # Part 2: Understanding execution flow with a counter example
    demonstrate_execution_flow()

    # Part 3: Graph with multiple fields showing selective updates
    demonstrate_multi_field_graph()

    print("\n" + "=" * 60)
    print("  Exercise 15 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. StateGraph uses TypedDict to define the state schema")
    print("  2. Nodes are functions that receive state and return partial updates")
    print("  3. Edges define the execution order between nodes")
    print("  4. START and END mark the entry and exit points")
    print("  5. compile() turns the builder into a runnable graph")
    print("  6. invoke() executes the graph and returns the final state")
    print("  7. Each node only updates the fields it returns")
    print("  8. The graph handles state merging automatically")
    print()


if __name__ == "__main__":
    main()
