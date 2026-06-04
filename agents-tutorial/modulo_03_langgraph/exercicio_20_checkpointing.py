"""
Module 3 - Exercise 20: Checkpointing
=======================================
Learn how to save and restore graph state using LangGraph's checkpointing
system. Checkpointing enables persistence, session management, and the
ability to pause and resume long-running workflows.

Concepts covered:
- Using MemorySaver for saving and restoring graph state
- Using thread_id for session management
- Resuming a graph from a saved checkpoint
- Inspecting checkpoint history
- Managing multiple concurrent sessions

Example: A multi-step data processing pipeline that can be paused and resumed.
"""

import sys
sys.path.append('..')
from config import load_config

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class DataPipelineState(TypedDict):
    """State for a multi-step data processing pipeline.

    Fields:
        data: The raw data being processed
        cleaned_data: Data after cleaning step
        transformed_data: Data after transformation step
        validated: Whether data passed validation
        enriched_data: Data after enrichment step
        final_output: Final processed result
        current_step: Name of the current processing step
        error_message: Error message if any step fails
        steps_completed: List of completed step names
        metadata: Additional metadata about the processing
    """
    data: str
    cleaned_data: str
    transformed_data: str
    validated: bool
    enriched_data: str
    final_output: str
    current_step: str
    error_message: str
    steps_completed: list[str]
    metadata: dict


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def ingest_data(state: DataPipelineState) -> dict:
    """Step 1: Ingest and validate raw data format."""
    data = state["data"]
    log = f"[ingest] Received data: '{data[:50]}{'...' if len(data) > 50 else ''}'"
    print(f"  -> {log}")

    return {
        "current_step": "ingest",
        "steps_completed": state.get("steps_completed", []) + ["ingest"],
        "metadata": {**state.get("metadata", {}), "original_length": len(data)},
    }


def clean_data(state: DataPipelineState) -> dict:
    """Step 2: Clean the data (remove extra whitespace, normalize)."""
    data = state["data"]

    # Simulate cleaning: strip, normalize whitespace, remove special chars
    cleaned = " ".join(data.split())
    cleaned = cleaned.strip()

    log = f"[clean] Cleaned data: '{cleaned[:50]}{'...' if len(cleaned) > 50 else ''}'"
    print(f"  -> {log}")

    return {
        "cleaned_data": cleaned,
        "current_step": "clean",
        "steps_completed": state["steps_completed"] + ["clean"],
        "metadata": {**state["metadata"], "cleaned_length": len(cleaned)},
    }


def transform_data(state: DataPipelineState) -> dict:
    """Step 3: Transform the data (convert to uppercase, add structure)."""
    cleaned = state["cleaned_data"]

    # Simulate transformation: uppercase and add structure markers
    transformed = f"[PROCESSED] {cleaned.upper()} [END]"

    log = f"[transform] Transformed: '{transformed[:50]}{'...' if len(transformed) > 50 else ''}'"
    print(f"  -> {log}")

    return {
        "transformed_data": transformed,
        "current_step": "transform",
        "steps_completed": state["steps_completed"] + ["transform"],
    }


def validate_data(state: DataPipelineState) -> dict:
    """Step 4: Validate the transformed data meets quality criteria."""
    transformed = state["transformed_data"]

    # Validation rules: must have markers and minimum length
    is_valid = (
        transformed.startswith("[PROCESSED]")
        and transformed.endswith("[END]")
        and len(transformed) > 20
    )

    status = "PASSED" if is_valid else "FAILED"
    log = f"[validate] Validation {status} (length={len(transformed)}, has_markers={is_valid})"
    print(f"  -> {log}")

    return {
        "validated": is_valid,
        "current_step": "validate",
        "steps_completed": state["steps_completed"] + ["validate"],
        "error_message": "" if is_valid else "Validation failed: missing markers or too short",
    }


def enrich_data(state: DataPipelineState) -> dict:
    """Step 5: Enrich the data with additional metadata."""
    transformed = state["transformed_data"]

    # Add enrichment: timestamp-like marker and word count
    words = len(transformed.split())
    enriched = f"{transformed} | words={words} | status=enriched"

    log = f"[enrich] Added metadata: words={words}"
    print(f"  -> {log}")

    return {
        "enriched_data": enriched,
        "current_step": "enrich",
        "steps_completed": state["steps_completed"] + ["enrich"],
    }


def finalize_data(state: DataPipelineState) -> dict:
    """Step 6: Produce the final output."""
    enriched = state["enriched_data"]
    steps = state["steps_completed"]

    final = f"FINAL OUTPUT: {enriched} | pipeline_steps={len(steps) + 1}"

    log = f"[finalize] Pipeline complete. Total steps: {len(steps) + 1}"
    print(f"  -> {log}")

    return {
        "final_output": final,
        "current_step": "finalize",
        "steps_completed": steps + ["finalize"],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------

def route_after_validation(state: DataPipelineState) -> str:
    """Route based on validation result."""
    if state["validated"]:
        return "enrich"
    else:
        return END


# ---------------------------------------------------------------------------
# Part 1: Basic Checkpointing with MemorySaver
# ---------------------------------------------------------------------------

def demonstrate_basic_checkpointing():
    """Show how MemorySaver saves state at each step of the graph."""
    print("\n" + "=" * 60)
    print("PART 1: Basic Checkpointing with MemorySaver")
    print("=" * 60)
    print()
    print("MemorySaver automatically saves the graph state after each node.")
    print("This creates a checkpoint history that you can inspect and restore from.")
    print()
    print("Key concepts:")
    print("  - Each graph invocation with a thread_id creates checkpoints")
    print("  - Checkpoints are saved after each node completes")
    print("  - You can inspect the full history of state changes")
    print()

    # Create checkpointer and graph
    checkpointer = MemorySaver()

    builder = StateGraph(DataPipelineState)
    builder.add_node("ingest", ingest_data)
    builder.add_node("clean", clean_data)
    builder.add_node("transform", transform_data)
    builder.add_node("validate", validate_data)
    builder.add_node("enrich", enrich_data)
    builder.add_node("finalize", finalize_data)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "clean")
    builder.add_edge("clean", "transform")
    builder.add_edge("transform", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"enrich": "enrich", END: END},
    )
    builder.add_edge("enrich", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer)

    # Run the pipeline with a thread_id
    config = {"configurable": {"thread_id": "pipeline-001"}}

    initial_state = {
        "data": "  Hello   World   from   the   data   pipeline!  ",
        "cleaned_data": "",
        "transformed_data": "",
        "validated": False,
        "enriched_data": "",
        "final_output": "",
        "current_step": "",
        "error_message": "",
        "steps_completed": [],
        "metadata": {},
    }

    print("  Running pipeline with thread_id='pipeline-001'...")
    print()

    result = graph.invoke(initial_state, config)

    print()
    print("  Pipeline completed!")
    print(f"  Steps completed: {result['steps_completed']}")
    print(f"  Final output: {result['final_output'][:80]}...")
    print()

    # Inspect checkpoint history
    print("Step 2: Inspecting checkpoint history")
    print("-" * 40)
    print()

    # Get all checkpoints for this thread
    checkpoints = list(graph.get_state_history(config))

    print(f"  Total checkpoints saved: {len(checkpoints)}")
    print()
    print("  Checkpoint history (most recent first):")
    for i, checkpoint in enumerate(checkpoints):
        step = checkpoint.values.get("current_step", "initial")
        steps_done = checkpoint.values.get("steps_completed", [])
        print(f"    [{i}] Step: '{step}' | Completed: {steps_done}")

    print()
    print("  Each checkpoint captures the FULL state at that point in time.")
    print("  You can restore to any checkpoint to resume from that point.")

    return graph, checkpointer


# ---------------------------------------------------------------------------
# Part 2: Session Management with thread_id
# ---------------------------------------------------------------------------

def demonstrate_session_management():
    """Show how thread_id enables multiple independent sessions."""
    print("\n" + "=" * 60)
    print("PART 2: Session Management with thread_id")
    print("=" * 60)
    print()
    print("Each thread_id represents an independent execution session.")
    print("Multiple sessions can run concurrently without interfering.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(DataPipelineState)
    builder.add_node("ingest", ingest_data)
    builder.add_node("clean", clean_data)
    builder.add_node("transform", transform_data)
    builder.add_node("validate", validate_data)
    builder.add_node("enrich", enrich_data)
    builder.add_node("finalize", finalize_data)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "clean")
    builder.add_edge("clean", "transform")
    builder.add_edge("transform", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"enrich": "enrich", END: END},
    )
    builder.add_edge("enrich", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer)

    # Session 1: Process dataset A
    print("  Session 1 (thread_id='session-A'): Processing dataset A")
    print("-" * 40)

    config_a = {"configurable": {"thread_id": "session-A"}}
    state_a = {
        "data": "Dataset Alpha: important business metrics",
        "cleaned_data": "",
        "transformed_data": "",
        "validated": False,
        "enriched_data": "",
        "final_output": "",
        "current_step": "",
        "error_message": "",
        "steps_completed": [],
        "metadata": {"source": "database_A"},
    }

    result_a = graph.invoke(state_a, config_a)
    print()

    # Session 2: Process dataset B
    print("  Session 2 (thread_id='session-B'): Processing dataset B")
    print("-" * 40)

    config_b = {"configurable": {"thread_id": "session-B"}}
    state_b = {
        "data": "Dataset Beta: customer feedback analysis results",
        "cleaned_data": "",
        "transformed_data": "",
        "validated": False,
        "enriched_data": "",
        "final_output": "",
        "current_step": "",
        "error_message": "",
        "steps_completed": [],
        "metadata": {"source": "database_B"},
    }

    result_b = graph.invoke(state_b, config_b)
    print()

    # Show that sessions are independent
    print("  Comparing sessions:")
    print(f"    Session A final step: {result_a['current_step']}")
    print(f"    Session B final step: {result_b['current_step']}")
    print()

    # Retrieve state for each session independently
    state_snapshot_a = graph.get_state(config_a)
    state_snapshot_b = graph.get_state(config_b)

    print(f"    Session A data source: {state_snapshot_a.values['metadata'].get('source')}")
    print(f"    Session B data source: {state_snapshot_b.values['metadata'].get('source')}")
    print()
    print("  Sessions are completely independent - each has its own state history.")


# ---------------------------------------------------------------------------
# Part 3: Pause and Resume from Checkpoint
# ---------------------------------------------------------------------------

def demonstrate_pause_and_resume():
    """Show how to pause a pipeline and resume from a checkpoint."""
    print("\n" + "=" * 60)
    print("PART 3: Pause and Resume from Checkpoint")
    print("=" * 60)
    print()
    print("Using interrupts with checkpointing allows you to:")
    print("  1. Pause a long-running pipeline at any point")
    print("  2. Inspect the current state")
    print("  3. Optionally modify the state")
    print("  4. Resume execution from where it left off")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(DataPipelineState)
    builder.add_node("ingest", ingest_data)
    builder.add_node("clean", clean_data)
    builder.add_node("transform", transform_data)
    builder.add_node("validate", validate_data)
    builder.add_node("enrich", enrich_data)
    builder.add_node("finalize", finalize_data)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "clean")
    builder.add_edge("clean", "transform")
    builder.add_edge("transform", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"enrich": "enrich", END: END},
    )
    builder.add_edge("enrich", "finalize")
    builder.add_edge("finalize", END)

    # Compile with interrupt_after transform (pause after transformation)
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["transform"],
    )

    config = {"configurable": {"thread_id": "pausable-pipeline"}}

    initial_state = {
        "data": "Critical report data that needs careful processing",
        "cleaned_data": "",
        "transformed_data": "",
        "validated": False,
        "enriched_data": "",
        "final_output": "",
        "current_step": "",
        "error_message": "",
        "steps_completed": [],
        "metadata": {"priority": "high"},
    }

    # Phase 1: Run until pause point
    print("  Phase 1: Running pipeline until transform completes...")
    print()

    partial_result = graph.invoke(initial_state, config)

    print()
    print("  Pipeline PAUSED after 'transform' step.")
    print(f"  Steps completed so far: {partial_result['steps_completed']}")
    print(f"  Current step: {partial_result['current_step']}")
    print(f"  Transformed data: {partial_result['transformed_data'][:60]}...")
    print()

    # Inspect the saved state
    print("  Inspecting saved checkpoint state:")
    current_state = graph.get_state(config)
    print(f"    Next nodes to execute: {current_state.next}")
    print(f"    Steps completed: {current_state.values['steps_completed']}")
    print()

    # Phase 2: Resume execution
    print("  Phase 2: Resuming pipeline from checkpoint...")
    print()

    # Resume by invoking with None (continues from last checkpoint)
    final_result = graph.invoke(None, config)

    print()
    print("  Pipeline RESUMED and completed!")
    print(f"  All steps completed: {final_result['steps_completed']}")
    print(f"  Final output: {final_result['final_output'][:80]}...")


# ---------------------------------------------------------------------------
# Part 4: Inspecting Checkpoint History
# ---------------------------------------------------------------------------

def demonstrate_checkpoint_inspection():
    """Show how to inspect and navigate checkpoint history."""
    print("\n" + "=" * 60)
    print("PART 4: Inspecting Checkpoint History")
    print("=" * 60)
    print()
    print("LangGraph stores the complete history of state transitions.")
    print("You can inspect any past state, compare states, and even")
    print("resume from a historical checkpoint.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(DataPipelineState)
    builder.add_node("ingest", ingest_data)
    builder.add_node("clean", clean_data)
    builder.add_node("transform", transform_data)
    builder.add_node("validate", validate_data)
    builder.add_node("enrich", enrich_data)
    builder.add_node("finalize", finalize_data)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "clean")
    builder.add_edge("clean", "transform")
    builder.add_edge("transform", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"enrich": "enrich", END: END},
    )
    builder.add_edge("enrich", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "history-demo"}}

    initial_state = {
        "data": "Sample data for checkpoint history demonstration",
        "cleaned_data": "",
        "transformed_data": "",
        "validated": False,
        "enriched_data": "",
        "final_output": "",
        "current_step": "",
        "error_message": "",
        "steps_completed": [],
        "metadata": {},
    }

    print("  Running full pipeline to generate checkpoint history...")
    print()

    result = graph.invoke(initial_state, config)

    print()
    print("  Pipeline complete. Now inspecting checkpoint history:")
    print("-" * 40)
    print()

    # Get full history
    history = list(graph.get_state_history(config))

    print(f"  Total checkpoints: {len(history)}")
    print()

    for i, state_snapshot in enumerate(history):
        step = state_snapshot.values.get("current_step", "(initial)")
        steps_done = state_snapshot.values.get("steps_completed", [])
        checkpoint_id = state_snapshot.config.get("configurable", {}).get("checkpoint_id", "N/A")

        print(f"  Checkpoint {i}:")
        print(f"    checkpoint_id: {checkpoint_id[:20]}..." if len(str(checkpoint_id)) > 20 else f"    checkpoint_id: {checkpoint_id}")
        print(f"    current_step: {step}")
        print(f"    steps_completed: {steps_done}")
        print(f"    next: {state_snapshot.next}")
        print()

    # Show how to get state at a specific checkpoint
    print("  Getting state at a specific checkpoint:")
    print("-" * 40)
    if len(history) > 2:
        # Get an intermediate checkpoint
        mid_checkpoint = history[len(history) // 2]
        mid_config = mid_checkpoint.config
        print(f"  Retrieving state from checkpoint at step: {mid_checkpoint.values.get('current_step', 'N/A')}")
        print(f"  Steps completed at that point: {mid_checkpoint.values.get('steps_completed', [])}")
        print()
        print("  This allows you to 'time travel' through the pipeline execution")
        print("  and inspect the state at any point in the history.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all checkpointing demonstrations."""
    print("=" * 60)
    print("  LangGraph: Checkpointing")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates checkpointing and state persistence.")
    print("No LLM calls are needed - we focus on checkpoint mechanics.")

    # Part 1: Basic checkpointing
    demonstrate_basic_checkpointing()

    # Part 2: Session management with thread_id
    demonstrate_session_management()

    # Part 3: Pause and resume
    demonstrate_pause_and_resume()

    # Part 4: Checkpoint history inspection
    demonstrate_checkpoint_inspection()

    print("\n" + "=" * 60)
    print("  Exercise 20 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. MemorySaver saves graph state after each node execution")
    print("  2. thread_id identifies independent execution sessions")
    print("  3. Multiple sessions can run concurrently without interference")
    print("  4. graph.get_state(config) retrieves the current state snapshot")
    print("  5. graph.get_state_history(config) returns all checkpoints")
    print("  6. Interrupts + checkpoints enable pause/resume workflows")
    print("  7. graph.invoke(None, config) resumes from the last checkpoint")
    print("  8. Each checkpoint has a unique checkpoint_id for identification")
    print("  9. You can inspect historical states for debugging and auditing")
    print()


if __name__ == "__main__":
    main()
