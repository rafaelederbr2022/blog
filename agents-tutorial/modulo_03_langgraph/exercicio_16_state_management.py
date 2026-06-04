"""
Module 3 - Exercise 16: State Management in LangGraph
======================================================
Learn how state is passed between nodes, modified, and accumulated
in a LangGraph StateGraph. This exercise covers advanced state patterns
including Annotated types with reducers.

Concepts covered:
- How state flows between nodes and gets modified
- Annotated types with reducer functions (e.g., add_messages)
- State accumulation across nodes (append vs replace)
- Reading and writing specific state fields in each node
- Custom reducer functions for list accumulation
- Multi-step data enrichment pipeline example

Example: A multi-step data enrichment pipeline where each node
adds information to the state.
"""

import sys
sys.path.append('..')
from config import load_config

import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ---------------------------------------------------------------------------
# Part 1: State Passing Between Nodes
# ---------------------------------------------------------------------------

def demonstrate_state_passing():
    """Show how state is passed between nodes and modified."""
    print("\n" + "=" * 60)
    print("PART 1: How State Passes Between Nodes")
    print("=" * 60)
    print()
    print("When a node returns a dict, LangGraph MERGES it with the current state.")
    print("This means:")
    print("  - Returned fields OVERWRITE existing values (for simple types)")
    print("  - Fields NOT returned remain unchanged")
    print("  - The next node receives the MERGED state")
    print()

    class PipelineState(TypedDict):
        data: str
        stage: str
        modifications: int

    def stage_one(state: PipelineState) -> dict:
        """First stage: uppercase the data."""
        print(f"  [stage_one] Received state:")
        print(f"    data='{state['data']}', stage='{state['stage']}', modifications={state['modifications']}")
        new_data = state["data"].upper()
        print(f"  [stage_one] Returning: data='{new_data}', stage='processed', modifications={state['modifications'] + 1}")
        return {
            "data": new_data,
            "stage": "processed",
            "modifications": state["modifications"] + 1,
        }

    def stage_two(state: PipelineState) -> dict:
        """Second stage: add prefix."""
        print(f"  [stage_two] Received state:")
        print(f"    data='{state['data']}', stage='{state['stage']}', modifications={state['modifications']}")
        new_data = f"[RESULT] {state['data']}"
        print(f"  [stage_two] Returning: data='{new_data}', stage='formatted', modifications={state['modifications'] + 1}")
        return {
            "data": new_data,
            "stage": "formatted",
            "modifications": state["modifications"] + 1,
        }

    # Build graph
    builder = StateGraph(PipelineState)
    builder.add_node("stage_one", stage_one)
    builder.add_node("stage_two", stage_two)
    builder.add_edge(START, "stage_one")
    builder.add_edge("stage_one", "stage_two")
    builder.add_edge("stage_two", END)
    graph = builder.compile()

    print("Executing graph with initial state:")
    print("-" * 40)
    initial = {"data": "hello world", "stage": "raw", "modifications": 0}
    print(f"  Initial: data='{initial['data']}', stage='{initial['stage']}', modifications={initial['modifications']}")
    print()

    result = graph.invoke(initial)

    print()
    print("Final state:")
    print(f"  data='{result['data']}'")
    print(f"  stage='{result['stage']}'")
    print(f"  modifications={result['modifications']}")
    print()
    print("Each node received the merged state from the previous node.")
    print("The 'modifications' counter shows how state accumulates.")


# ---------------------------------------------------------------------------
# Part 2: Annotated Types with Reducers
# ---------------------------------------------------------------------------

def demonstrate_annotated_reducers():
    """Show how Annotated types with reducers enable state accumulation."""
    print("\n" + "=" * 60)
    print("PART 2: Annotated Types with Reducers")
    print("=" * 60)
    print()
    print("By default, returning a field REPLACES its value.")
    print("But with Annotated[type, reducer], you can define HOW values combine.")
    print()
    print("Common reducers:")
    print("  - operator.add: concatenates lists (append behavior)")
    print("  - add_messages: intelligently merges message lists")
    print("  - Custom functions: any (old, new) -> combined logic")
    print()

    # Using operator.add as a reducer for list accumulation
    class AccumulatorState(TypedDict):
        # This field uses operator.add as reducer:
        # When a node returns {"items": [new_item]}, it APPENDS to the list
        # instead of replacing it
        items: Annotated[list[str], operator.add]
        # This field has no reducer - it gets REPLACED each time
        current_step: str

    def collect_fruits(state: AccumulatorState) -> dict:
        """Add fruits to the items list."""
        print(f"  [collect_fruits] Current items: {state['items']}")
        print(f"  [collect_fruits] Adding: ['apple', 'banana']")
        return {
            "items": ["apple", "banana"],  # These get APPENDED due to reducer
            "current_step": "fruits_collected",
        }

    def collect_vegetables(state: AccumulatorState) -> dict:
        """Add vegetables to the items list."""
        print(f"  [collect_vegetables] Current items: {state['items']}")
        print(f"  [collect_vegetables] Adding: ['carrot', 'broccoli']")
        return {
            "items": ["carrot", "broccoli"],  # These get APPENDED due to reducer
            "current_step": "vegetables_collected",
        }

    def collect_grains(state: AccumulatorState) -> dict:
        """Add grains to the items list."""
        print(f"  [collect_grains] Current items: {state['items']}")
        print(f"  [collect_grains] Adding: ['rice', 'wheat']")
        return {
            "items": ["rice", "wheat"],  # These get APPENDED due to reducer
            "current_step": "grains_collected",
        }

    # Build graph
    builder = StateGraph(AccumulatorState)
    builder.add_node("fruits", collect_fruits)
    builder.add_node("vegetables", collect_vegetables)
    builder.add_node("grains", collect_grains)

    builder.add_edge(START, "fruits")
    builder.add_edge("fruits", "vegetables")
    builder.add_edge("vegetables", "grains")
    builder.add_edge("grains", END)

    graph = builder.compile()

    print("Executing accumulator graph:")
    print("-" * 40)
    print("  Each node returns a list that gets APPENDED (not replaced)")
    print("  because 'items' uses Annotated[list, operator.add] as reducer.")
    print()

    result = graph.invoke({"items": [], "current_step": "start"})

    print()
    print("Final state:")
    print(f"  items: {result['items']}")
    print(f"  current_step: '{result['current_step']}'")
    print()
    print("  Notice: 'items' accumulated ALL values from ALL nodes!")
    print("  Without the reducer, only the last node's items would remain.")
    print()
    print("  'current_step' has NO reducer, so it was REPLACED each time.")
    print("  Only the last value ('grains_collected') survived.")


# ---------------------------------------------------------------------------
# Part 3: add_messages Reducer for Chat Messages
# ---------------------------------------------------------------------------

def demonstrate_add_messages():
    """Show the add_messages reducer for managing conversation state."""
    print("\n" + "=" * 60)
    print("PART 3: add_messages Reducer for Chat Messages")
    print("=" * 60)
    print()
    print("The add_messages reducer is specifically designed for message lists.")
    print("It appends new messages to the existing list, which is the standard")
    print("pattern for building conversational agents with LangGraph.")
    print()

    class ChatState(TypedDict):
        # add_messages appends new messages to the list
        messages: Annotated[list, add_messages]
        context: str

    def greet_node(state: ChatState) -> dict:
        """Add a greeting message to the conversation."""
        print(f"  [greet_node] Messages so far: {len(state['messages'])}")
        greeting = AIMessage(content="Hello! I'm your assistant. How can I help?")
        print(f"  [greet_node] Adding AI greeting message")
        return {"messages": [greeting]}

    def process_query_node(state: ChatState) -> dict:
        """Process the user's query and add a response."""
        print(f"  [process_query_node] Messages so far: {len(state['messages'])}")
        # Read the last message (simulating processing)
        last_msg = state["messages"][-1]
        print(f"  [process_query_node] Last message: '{last_msg.content}'")

        response = AIMessage(
            content=f"I understand you're asking about: '{last_msg.content}'. "
                    f"Context: {state['context']}"
        )
        print(f"  [process_query_node] Adding AI response")
        return {"messages": [response]}

    # Build graph
    builder = StateGraph(ChatState)
    builder.add_node("greet", greet_node)
    builder.add_node("process", process_query_node)

    builder.add_edge(START, "greet")
    builder.add_edge("greet", "process")
    builder.add_edge("process", END)

    graph = builder.compile()

    print("Executing chat graph with add_messages reducer:")
    print("-" * 40)
    print()

    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is LangGraph?"),
        ],
        "context": "LangGraph documentation",
    }

    print(f"  Initial messages: {len(initial_state['messages'])}")
    for msg in initial_state["messages"]:
        print(f"    [{msg.__class__.__name__}] {msg.content}")
    print()

    result = graph.invoke(initial_state)

    print()
    print(f"  Final messages: {len(result['messages'])}")
    for msg in result["messages"]:
        print(f"    [{msg.__class__.__name__}] {msg.content[:80]}...")
    print()
    print("  The add_messages reducer appended new messages to the existing list.")
    print("  This is the standard pattern for conversational agents.")


# ---------------------------------------------------------------------------
# Part 4: Multi-Step Data Enrichment Pipeline
# ---------------------------------------------------------------------------

def demonstrate_data_enrichment():
    """Show a multi-step pipeline where each node enriches the state."""
    print("\n" + "=" * 60)
    print("PART 4: Multi-Step Data Enrichment Pipeline")
    print("=" * 60)
    print()
    print("Real-world example: Each node adds information to a user profile.")
    print("This demonstrates how state grows as it passes through the graph.")
    print()

    class UserProfileState(TypedDict):
        # Basic info (set initially)
        user_id: str
        username: str
        # Enriched fields (added by nodes)
        email_domain: str
        account_tier: str
        permissions: Annotated[list[str], operator.add]
        enrichment_log: Annotated[list[str], operator.add]

    def extract_domain(state: UserProfileState) -> dict:
        """Extract email domain from username (simulated)."""
        # Simulate extracting domain info
        domain = f"{state['username'].lower()}.company.com"
        log = f"Extracted domain: {domain}"
        print(f"  -> [extract_domain] {log}")
        return {
            "email_domain": domain,
            "enrichment_log": [log],
        }

    def determine_tier(state: UserProfileState) -> dict:
        """Determine account tier based on user_id (simulated)."""
        # Simulate tier determination
        user_num = hash(state["user_id"]) % 100
        tier = "premium" if user_num > 50 else "standard"
        log = f"Determined tier: {tier} (based on user_id hash)"
        print(f"  -> [determine_tier] {log}")
        return {
            "account_tier": tier,
            "enrichment_log": [log],
        }

    def assign_permissions(state: UserProfileState) -> dict:
        """Assign permissions based on account tier."""
        base_permissions = ["read", "write"]
        if state["account_tier"] == "premium":
            extra = ["admin", "export", "api_access"]
        else:
            extra = ["basic_api"]

        all_perms = base_permissions + extra
        log = f"Assigned permissions: {all_perms} (tier: {state['account_tier']})"
        print(f"  -> [assign_permissions] {log}")
        return {
            "permissions": all_perms,
            "enrichment_log": [log],
        }

    def finalize_profile(state: UserProfileState) -> dict:
        """Final node: log the completed enrichment."""
        log = f"Profile enrichment complete for user '{state['username']}'"
        print(f"  -> [finalize_profile] {log}")
        return {
            "enrichment_log": [log],
        }

    # Build the enrichment pipeline
    builder = StateGraph(UserProfileState)
    builder.add_node("extract_domain", extract_domain)
    builder.add_node("determine_tier", determine_tier)
    builder.add_node("assign_permissions", assign_permissions)
    builder.add_node("finalize", finalize_profile)

    builder.add_edge(START, "extract_domain")
    builder.add_edge("extract_domain", "determine_tier")
    builder.add_edge("determine_tier", "assign_permissions")
    builder.add_edge("assign_permissions", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile()

    # Execute the enrichment pipeline
    print("Executing data enrichment pipeline:")
    print("-" * 40)
    print()

    initial = {
        "user_id": "usr_12345",
        "username": "JohnDoe",
        "email_domain": "",
        "account_tier": "",
        "permissions": [],
        "enrichment_log": [],
    }

    print(f"  Initial state: user_id='{initial['user_id']}', username='{initial['username']}'")
    print(f"  (All enrichment fields are empty)")
    print()

    result = graph.invoke(initial)

    print()
    print("=" * 40)
    print("  ENRICHED USER PROFILE:")
    print("=" * 40)
    print(f"  user_id: {result['user_id']}")
    print(f"  username: {result['username']}")
    print(f"  email_domain: {result['email_domain']}")
    print(f"  account_tier: {result['account_tier']}")
    print(f"  permissions: {result['permissions']}")
    print()
    print("  Enrichment log (accumulated via reducer):")
    for i, entry in enumerate(result["enrichment_log"], 1):
        print(f"    {i}. {entry}")
    print()
    print("  Key observations:")
    print("  - 'permissions' and 'enrichment_log' used operator.add reducer")
    print("    so values ACCUMULATED across nodes")
    print("  - 'email_domain' and 'account_tier' used simple replacement")
    print("  - 'user_id' and 'username' were never modified (no node returned them)")


# ---------------------------------------------------------------------------
# Part 5: Custom Reducer Functions
# ---------------------------------------------------------------------------

def demonstrate_custom_reducer():
    """Show how to create custom reducer functions."""
    print("\n" + "=" * 60)
    print("PART 5: Custom Reducer Functions")
    print("=" * 60)
    print()
    print("You can define any function as a reducer.")
    print("A reducer takes (existing_value, new_value) and returns the combined result.")
    print()

    def max_reducer(existing: int, new: int) -> int:
        """Keep the maximum value seen so far."""
        return max(existing, new)

    def merge_dicts_reducer(existing: dict, new: dict) -> dict:
        """Merge dictionaries, with new values overwriting existing ones."""
        merged = {**existing, **new}
        return merged

    class MetricsState(TypedDict):
        # Custom reducer: always keep the maximum score
        max_score: Annotated[int, max_reducer]
        # Custom reducer: merge metadata dicts
        metadata: Annotated[dict, merge_dicts_reducer]
        # Standard reducer: accumulate list
        scores: Annotated[list[int], operator.add]

    def evaluator_a(state: MetricsState) -> dict:
        """First evaluator gives a score."""
        score = 75
        print(f"  [evaluator_a] Score: {score}")
        return {
            "max_score": score,
            "metadata": {"evaluator_a": "completed", "timestamp_a": "2024-01-01"},
            "scores": [score],
        }

    def evaluator_b(state: MetricsState) -> dict:
        """Second evaluator gives a higher score."""
        score = 92
        print(f"  [evaluator_b] Score: {score}")
        return {
            "max_score": score,
            "metadata": {"evaluator_b": "completed", "timestamp_b": "2024-01-02"},
            "scores": [score],
        }

    def evaluator_c(state: MetricsState) -> dict:
        """Third evaluator gives a lower score."""
        score = 68
        print(f"  [evaluator_c] Score: {score}")
        return {
            "max_score": score,
            "metadata": {"evaluator_c": "completed", "final": True},
            "scores": [score],
        }

    # Build graph
    builder = StateGraph(MetricsState)
    builder.add_node("eval_a", evaluator_a)
    builder.add_node("eval_b", evaluator_b)
    builder.add_node("eval_c", evaluator_c)

    builder.add_edge(START, "eval_a")
    builder.add_edge("eval_a", "eval_b")
    builder.add_edge("eval_b", "eval_c")
    builder.add_edge("eval_c", END)

    graph = builder.compile()

    print("Executing metrics pipeline with custom reducers:")
    print("-" * 40)
    print()

    result = graph.invoke({
        "max_score": 0,
        "metadata": {},
        "scores": [],
    })

    print()
    print("Final state:")
    print(f"  max_score: {result['max_score']} (max_reducer kept the highest: 92)")
    print(f"  scores: {result['scores']} (operator.add accumulated all)")
    print(f"  metadata: {result['metadata']} (merge_dicts combined all entries)")
    print()
    print("  Custom reducers give you full control over how state combines!")
    print("  - max_reducer: keeps only the highest value")
    print("  - merge_dicts_reducer: merges all dicts into one")
    print("  - operator.add: concatenates lists")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all state management demonstrations."""
    print("=" * 60)
    print("  LangGraph: State Management")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates state management patterns in LangGraph.")
    print("No LLM calls are needed - we focus on state flow and reducers.")

    # Part 1: How state passes between nodes
    demonstrate_state_passing()

    # Part 2: Annotated types with reducers
    demonstrate_annotated_reducers()

    # Part 3: add_messages reducer for chat
    demonstrate_add_messages()

    # Part 4: Multi-step data enrichment pipeline
    demonstrate_data_enrichment()

    # Part 5: Custom reducer functions
    demonstrate_custom_reducer()

    print("\n" + "=" * 60)
    print("  Exercise 16 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Nodes receive full state and return partial updates (dict)")
    print("  2. Without a reducer, returned values REPLACE existing ones")
    print("  3. Annotated[type, reducer] defines HOW values combine")
    print("  4. operator.add is the standard reducer for list accumulation")
    print("  5. add_messages is the standard reducer for chat message lists")
    print("  6. Custom reducers take (old, new) and return the combined value")
    print("  7. Different fields can use different reducers in the same state")
    print("  8. This enables powerful patterns like data enrichment pipelines")
    print()


if __name__ == "__main__":
    main()
