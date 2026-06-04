"""
Module 3 - Exercise 22: Long Running Workflows
================================================
Build workflows that simulate long-running operations and demonstrate how
LangGraph's checkpointing enables resuming after interruptions. This exercise
shows progress tracking, timeout handling, and retry logic for workflows
that take extended periods to complete.

Concepts covered:
- Simulating long-running operations with time.sleep
- Checkpointing enables resuming after system restarts
- Progress tracking through state updates
- Handling timeouts and retries in long-running nodes
- Multi-phase pipelines with progress percentage reporting

Example: A data migration pipeline with extract, transform, and load phases,
where each phase reports progress and can be resumed if interrupted.
"""

import sys
sys.path.append('..')
from config import load_config

import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class MigrationState(TypedDict):
    """State for a data migration pipeline with progress tracking.

    Fields:
        job_id: Unique identifier for this migration job
        source_system: Name of the source system
        target_system: Name of the target system
        total_records: Total number of records to migrate
        current_phase: Current phase (extract/transform/load/complete)
        extract_progress: Percentage of extraction completed (0-100)
        transform_progress: Percentage of transformation completed (0-100)
        load_progress: Percentage of loading completed (0-100)
        records_extracted: Number of records extracted so far
        records_transformed: Number of records transformed so far
        records_loaded: Number of records loaded so far
        errors: List of errors encountered during migration
        retry_count: Number of retries attempted
        max_retries: Maximum number of retries allowed
        is_complete: Whether the migration is fully complete
        elapsed_seconds: Total elapsed time in seconds
        steps_log: Log of all steps executed
    """
    job_id: str
    source_system: str
    target_system: str
    total_records: int
    current_phase: str
    extract_progress: int
    transform_progress: int
    load_progress: int
    records_extracted: int
    records_transformed: int
    records_loaded: int
    errors: list[str]
    retry_count: int
    max_retries: int
    is_complete: bool
    elapsed_seconds: float
    steps_log: list[str]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def simulate_work(duration: float, description: str) -> None:
    """Simulate a long-running operation with a short sleep.

    In a real application, this would be actual I/O, network calls, etc.
    We use very short sleeps (0.1s) for demonstration purposes.
    """
    time.sleep(duration)


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def initialize_migration(state: MigrationState) -> dict:
    """Initialize the migration job and validate configuration."""
    job_id = state["job_id"]
    source = state["source_system"]
    target = state["target_system"]
    total = state["total_records"]

    log_entry = (
        f"[init] Migration job '{job_id}' initialized: "
        f"{source} -> {target} ({total} records)"
    )
    print(f"  -> {log_entry}")

    simulate_work(0.1, "Validating connection to source and target systems")

    print(f"     Connections validated. Ready to begin extraction.")

    return {
        "current_phase": "initialized",
        "steps_log": state.get("steps_log", []) + [log_entry],
    }


def extract_phase(state: MigrationState) -> dict:
    """Extract data from the source system with progress tracking.

    Simulates extracting records in batches, updating progress percentage.
    """
    total = state["total_records"]
    already_extracted = state.get("records_extracted", 0)

    log_entry = f"[extract] Starting extraction phase (already extracted: {already_extracted}/{total})"
    print(f"  -> {log_entry}")

    # Simulate batch extraction with progress updates
    batch_size = total // 4  # Process in 4 batches
    extracted = already_extracted

    for batch_num in range(1, 5):
        if extracted >= total:
            break

        batch_end = min(extracted + batch_size, total)
        simulate_work(0.05, f"Extracting batch {batch_num}")

        extracted = batch_end
        progress = int((extracted / total) * 100)
        print(f"     Extract progress: {progress}% ({extracted}/{total} records)")

    final_progress = int((extracted / total) * 100)
    log_complete = f"[extract] Extraction complete: {extracted}/{total} records ({final_progress}%)"
    print(f"  -> {log_complete}")

    return {
        "current_phase": "extract",
        "extract_progress": final_progress,
        "records_extracted": extracted,
        "steps_log": state["steps_log"] + [log_entry, log_complete],
    }


def transform_phase(state: MigrationState) -> dict:
    """Transform extracted data with progress tracking.

    Simulates data transformation (schema mapping, validation, enrichment).
    """
    total = state["records_extracted"]
    already_transformed = state.get("records_transformed", 0)

    log_entry = f"[transform] Starting transformation phase ({already_transformed}/{total} already done)"
    print(f"  -> {log_entry}")

    # Simulate transformation in batches
    batch_size = total // 4
    transformed = already_transformed

    for batch_num in range(1, 5):
        if transformed >= total:
            break

        batch_end = min(transformed + batch_size, total)
        simulate_work(0.05, f"Transforming batch {batch_num}")

        transformed = batch_end
        progress = int((transformed / total) * 100)
        print(f"     Transform progress: {progress}% ({transformed}/{total} records)")

    final_progress = int((transformed / total) * 100)
    log_complete = f"[transform] Transformation complete: {transformed}/{total} records ({final_progress}%)"
    print(f"  -> {log_complete}")

    return {
        "current_phase": "transform",
        "transform_progress": final_progress,
        "records_transformed": transformed,
        "steps_log": state["steps_log"] + [log_entry, log_complete],
    }


def load_phase(state: MigrationState) -> dict:
    """Load transformed data into the target system with progress tracking.

    Simulates writing records to the target system in batches.
    """
    total = state["records_transformed"]
    already_loaded = state.get("records_loaded", 0)

    log_entry = f"[load] Starting load phase ({already_loaded}/{total} already loaded)"
    print(f"  -> {log_entry}")

    # Simulate loading in batches
    batch_size = total // 4
    loaded = already_loaded

    for batch_num in range(1, 5):
        if loaded >= total:
            break

        batch_end = min(loaded + batch_size, total)
        simulate_work(0.05, f"Loading batch {batch_num}")

        loaded = batch_end
        progress = int((loaded / total) * 100)
        print(f"     Load progress: {progress}% ({loaded}/{total} records)")

    final_progress = int((loaded / total) * 100)
    log_complete = f"[load] Load complete: {loaded}/{total} records ({final_progress}%)"
    print(f"  -> {log_complete}")

    return {
        "current_phase": "load",
        "load_progress": final_progress,
        "records_loaded": loaded,
        "steps_log": state["steps_log"] + [log_entry, log_complete],
    }


def finalize_migration(state: MigrationState) -> dict:
    """Finalize the migration and produce summary report."""
    job_id = state["job_id"]
    total = state["total_records"]
    loaded = state["records_loaded"]
    errors = state.get("errors", [])

    log_entry = (
        f"[finalize] Migration '{job_id}' complete! "
        f"Records: {loaded}/{total} migrated, {len(errors)} errors"
    )
    print(f"  -> {log_entry}")

    return {
        "current_phase": "complete",
        "is_complete": True,
        "steps_log": state["steps_log"] + [log_entry],
    }


# ---------------------------------------------------------------------------
# Timeout and Retry Nodes
# ---------------------------------------------------------------------------

class TimeoutState(TypedDict):
    """State for demonstrating timeout and retry handling.

    Fields:
        operation_name: Name of the operation being attempted
        attempt_number: Current attempt number
        max_attempts: Maximum number of attempts
        timeout_seconds: Timeout threshold in seconds
        operation_duration: How long the operation actually takes
        success: Whether the operation succeeded
        error_message: Error message if operation failed
        steps_log: Log of all steps
    """
    operation_name: str
    attempt_number: int
    max_attempts: int
    timeout_seconds: float
    operation_duration: float
    success: bool
    error_message: str
    steps_log: list[str]


def attempt_operation(state: TimeoutState) -> dict:
    """Attempt a long-running operation with timeout checking.

    Simulates an operation that may exceed its timeout threshold.
    Uses decreasing duration on retries to simulate eventual success.
    """
    attempt = state["attempt_number"] + 1
    timeout = state["timeout_seconds"]
    operation = state["operation_name"]

    # Simulate decreasing duration on each retry (system recovers)
    # First attempt takes full duration, subsequent attempts are faster
    actual_duration = state["operation_duration"] / attempt

    log_entry = (
        f"[attempt] {operation} - Attempt {attempt}/{state['max_attempts']} "
        f"(timeout: {timeout}s, estimated duration: {actual_duration:.2f}s)"
    )
    print(f"  -> {log_entry}")

    # Simulate the operation
    simulate_work(min(actual_duration, 0.1), f"Attempt {attempt}")

    # Check if operation would have exceeded timeout
    timed_out = actual_duration > timeout

    if timed_out:
        error = f"Operation timed out after {timeout}s (took {actual_duration:.2f}s)"
        print(f"     TIMEOUT: {error}")
        return {
            "attempt_number": attempt,
            "success": False,
            "error_message": error,
            "steps_log": state.get("steps_log", []) + [log_entry, f"  TIMEOUT: {error}"],
        }
    else:
        print(f"     SUCCESS: Completed in {actual_duration:.2f}s (within {timeout}s timeout)")
        return {
            "attempt_number": attempt,
            "success": True,
            "error_message": "",
            "steps_log": state.get("steps_log", []) + [log_entry, f"  SUCCESS in {actual_duration:.2f}s"],
        }


def handle_timeout(state: TimeoutState) -> dict:
    """Handle a timeout by preparing for retry with backoff."""
    attempt = state["attempt_number"]
    max_attempts = state["max_attempts"]

    # Exponential backoff: wait longer between retries
    backoff_time = 2 ** (attempt - 1) * 0.5  # 0.5s, 1s, 2s, 4s...

    log_entry = (
        f"[retry] Preparing retry {attempt + 1}/{max_attempts}. "
        f"Backoff: {backoff_time:.1f}s"
    )
    print(f"  -> {log_entry}")

    # Simulate backoff wait (very short for demo)
    simulate_work(0.05, f"Backoff wait")

    return {
        "steps_log": state["steps_log"] + [log_entry],
    }


def report_failure(state: TimeoutState) -> dict:
    """Report that all retry attempts have been exhausted."""
    operation = state["operation_name"]
    attempts = state["attempt_number"]

    log_entry = (
        f"[failure] {operation} FAILED after {attempts} attempts. "
        f"Last error: {state['error_message']}"
    )
    print(f"  -> {log_entry}")

    return {
        "steps_log": state["steps_log"] + [log_entry],
    }


def report_success(state: TimeoutState) -> dict:
    """Report successful completion of the operation."""
    operation = state["operation_name"]
    attempts = state["attempt_number"]

    log_entry = f"[success] {operation} completed successfully on attempt {attempts}"
    print(f"  -> {log_entry}")

    return {
        "steps_log": state["steps_log"] + [log_entry],
    }


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_after_attempt(state: TimeoutState) -> str:
    """Route based on operation result: success, retry, or failure."""
    if state["success"]:
        return "report_success"
    elif state["attempt_number"] >= state["max_attempts"]:
        return "report_failure"
    else:
        return "handle_timeout"


# ---------------------------------------------------------------------------
# Part 1: Data Migration Pipeline with Progress Tracking
# ---------------------------------------------------------------------------

def demonstrate_migration_pipeline():
    """Show a multi-phase migration pipeline with progress tracking."""
    print("\n" + "=" * 60)
    print("PART 1: Data Migration Pipeline with Progress Tracking")
    print("=" * 60)
    print()
    print("A data migration pipeline with three phases:")
    print("  1. Extract - Pull data from source system")
    print("  2. Transform - Convert data to target format")
    print("  3. Load - Write data to target system")
    print()
    print("Each phase reports progress percentage through state updates.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(MigrationState)
    builder.add_node("initialize", initialize_migration)
    builder.add_node("extract", extract_phase)
    builder.add_node("transform", transform_phase)
    builder.add_node("load", load_phase)
    builder.add_node("finalize", finalize_migration)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "extract")
    builder.add_edge("extract", "transform")
    builder.add_edge("transform", "load")
    builder.add_edge("load", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "migration-001"}}

    initial_state = {
        "job_id": "MIG-2024-001",
        "source_system": "Legacy PostgreSQL",
        "target_system": "Cloud Datastore",
        "total_records": 1000,
        "current_phase": "pending",
        "extract_progress": 0,
        "transform_progress": 0,
        "load_progress": 0,
        "records_extracted": 0,
        "records_transformed": 0,
        "records_loaded": 0,
        "errors": [],
        "retry_count": 0,
        "max_retries": 3,
        "is_complete": False,
        "elapsed_seconds": 0.0,
        "steps_log": [],
    }

    print("  Starting migration: Legacy PostgreSQL -> Cloud Datastore")
    print("  Records to migrate: 1000")
    print()

    start_time = time.time()
    result = graph.invoke(initial_state, config)
    elapsed = time.time() - start_time

    print()
    print("  Migration Summary:")
    print(f"    Job ID: {result['job_id']}")
    print(f"    Phase: {result['current_phase']}")
    print(f"    Extract: {result['extract_progress']}%")
    print(f"    Transform: {result['transform_progress']}%")
    print(f"    Load: {result['load_progress']}%")
    print(f"    Records loaded: {result['records_loaded']}/{result['total_records']}")
    print(f"    Complete: {result['is_complete']}")
    print(f"    Elapsed: {elapsed:.2f}s")
    print()


# ---------------------------------------------------------------------------
# Part 2: Pause and Resume (Simulating System Restart)
# ---------------------------------------------------------------------------

def demonstrate_pause_and_resume():
    """Show how checkpointing enables resuming after interruption."""
    print("\n" + "=" * 60)
    print("PART 2: Pause and Resume (Simulating System Restart)")
    print("=" * 60)
    print()
    print("Long-running workflows can be interrupted (system crash, restart)")
    print("and resumed from the last checkpoint. This is critical for")
    print("operations that take hours or days to complete.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(MigrationState)
    builder.add_node("initialize", initialize_migration)
    builder.add_node("extract", extract_phase)
    builder.add_node("transform", transform_phase)
    builder.add_node("load", load_phase)
    builder.add_node("finalize", finalize_migration)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "extract")
    builder.add_edge("extract", "transform")
    builder.add_edge("transform", "load")
    builder.add_edge("load", "finalize")
    builder.add_edge("finalize", END)

    # Compile with interrupt_after extract phase (simulates interruption)
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["extract"],
    )

    config = {"configurable": {"thread_id": "migration-002"}}

    initial_state = {
        "job_id": "MIG-2024-002",
        "source_system": "On-Premise Oracle",
        "target_system": "AWS Aurora",
        "total_records": 500,
        "current_phase": "pending",
        "extract_progress": 0,
        "transform_progress": 0,
        "load_progress": 0,
        "records_extracted": 0,
        "records_transformed": 0,
        "records_loaded": 0,
        "errors": [],
        "retry_count": 0,
        "max_retries": 3,
        "is_complete": False,
        "elapsed_seconds": 0.0,
        "steps_log": [],
    }

    # Phase 1: Run until extraction completes (then pause)
    print("  Phase 1: Running migration until extraction completes...")
    print()

    partial_result = graph.invoke(initial_state, config)

    print()
    print("  === SYSTEM INTERRUPTED (simulated restart) ===")
    print(f"  State at interruption:")
    print(f"    Current phase: {partial_result['current_phase']}")
    print(f"    Extract progress: {partial_result['extract_progress']}%")
    print(f"    Records extracted: {partial_result['records_extracted']}")
    print(f"    Transform progress: {partial_result['transform_progress']}%")
    print(f"    Load progress: {partial_result['load_progress']}%")
    print()

    # Inspect checkpoint
    saved_state = graph.get_state(config)
    print(f"  Checkpoint saved. Next nodes: {saved_state.next}")
    print()

    # Phase 2: Resume from checkpoint (simulates system coming back online)
    print("  Phase 2: System back online. Resuming from checkpoint...")
    print()

    final_result = graph.invoke(None, config)

    print()
    print("  Migration resumed and completed!")
    print(f"    Final phase: {final_result['current_phase']}")
    print(f"    All records loaded: {final_result['records_loaded']}/{final_result['total_records']}")
    print(f"    Complete: {final_result['is_complete']}")
    print()
    print("  Key insight: The checkpoint preserved all progress from before")
    print("  the interruption. No data was re-extracted.")


# ---------------------------------------------------------------------------
# Part 3: Timeout and Retry Handling
# ---------------------------------------------------------------------------

def demonstrate_timeout_retry():
    """Show how to handle timeouts and retries in long-running nodes."""
    print("\n" + "=" * 60)
    print("PART 3: Timeout and Retry Handling")
    print("=" * 60)
    print()
    print("Long-running operations may time out. This demonstrates:")
    print("  - Detecting when an operation exceeds its timeout")
    print("  - Implementing retry logic with exponential backoff")
    print("  - Routing to success or failure based on retry results")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(TimeoutState)
    builder.add_node("attempt", attempt_operation)
    builder.add_node("handle_timeout", handle_timeout)
    builder.add_node("report_success", report_success)
    builder.add_node("report_failure", report_failure)

    builder.add_edge(START, "attempt")
    builder.add_conditional_edges(
        "attempt",
        route_after_attempt,
        {
            "report_success": "report_success",
            "handle_timeout": "handle_timeout",
            "report_failure": "report_failure",
        },
    )
    # After handling timeout, retry the operation
    builder.add_edge("handle_timeout", "attempt")
    builder.add_edge("report_success", END)
    builder.add_edge("report_failure", END)

    graph = builder.compile(checkpointer=checkpointer)

    # Scenario A: Operation succeeds after retries
    print("  Scenario A: Operation succeeds after retries")
    print("-" * 40)
    print()

    config_a = {"configurable": {"thread_id": "timeout-001"}}

    # Operation takes 3s but timeout is 2s. On retry 3 (duration/3=1s), it succeeds.
    state_a = {
        "operation_name": "Database backup",
        "attempt_number": 0,
        "max_attempts": 5,
        "timeout_seconds": 2.0,
        "operation_duration": 3.0,  # First attempt: 3s (timeout), second: 1.5s (timeout), third: 1s (success)
        "success": False,
        "error_message": "",
        "steps_log": [],
    }

    result_a = graph.invoke(state_a, config_a)
    print()
    print(f"  Result: success={result_a['success']}, attempts={result_a['attempt_number']}")
    print()

    # Scenario B: Operation fails after exhausting all retries
    print("  Scenario B: Operation fails after exhausting retries")
    print("-" * 40)
    print()

    config_b = {"configurable": {"thread_id": "timeout-002"}}

    # Operation takes 10s but timeout is 2s. Even at attempt 3 (10/3=3.3s), still too slow.
    state_b = {
        "operation_name": "External API sync",
        "attempt_number": 0,
        "max_attempts": 3,
        "timeout_seconds": 2.0,
        "operation_duration": 10.0,  # Always exceeds timeout even with retries
        "success": False,
        "error_message": "",
        "steps_log": [],
    }

    result_b = graph.invoke(state_b, config_b)
    print()
    print(f"  Result: success={result_b['success']}, attempts={result_b['attempt_number']}")
    print(f"  Error: {result_b['error_message']}")
    print()


# ---------------------------------------------------------------------------
# Part 4: Progress Monitoring Across Checkpoints
# ---------------------------------------------------------------------------

def demonstrate_progress_monitoring():
    """Show how to monitor progress of a long-running workflow via checkpoints."""
    print("\n" + "=" * 60)
    print("PART 4: Progress Monitoring via Checkpoints")
    print("=" * 60)
    print()
    print("In production, you can poll the checkpoint state to monitor")
    print("progress of long-running workflows without interrupting them.")
    print()
    print("This simulates checking progress at each phase boundary.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(MigrationState)
    builder.add_node("initialize", initialize_migration)
    builder.add_node("extract", extract_phase)
    builder.add_node("transform", transform_phase)
    builder.add_node("load", load_phase)
    builder.add_node("finalize", finalize_migration)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "extract")
    builder.add_edge("extract", "transform")
    builder.add_edge("transform", "load")
    builder.add_edge("load", "finalize")
    builder.add_edge("finalize", END)

    # Interrupt after each phase to simulate progress checks
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["initialize", "extract", "transform", "load"],
    )

    config = {"configurable": {"thread_id": "migration-003"}}

    initial_state = {
        "job_id": "MIG-2024-003",
        "source_system": "MongoDB Atlas",
        "target_system": "BigQuery",
        "total_records": 2000,
        "current_phase": "pending",
        "extract_progress": 0,
        "transform_progress": 0,
        "load_progress": 0,
        "records_extracted": 0,
        "records_transformed": 0,
        "records_loaded": 0,
        "errors": [],
        "retry_count": 0,
        "max_retries": 3,
        "is_complete": False,
        "elapsed_seconds": 0.0,
        "steps_log": [],
    }

    print("  Running migration with progress monitoring at each phase...")
    print()

    # Run phase by phase, checking progress at each checkpoint
    phases = ["initialize", "extract", "transform", "load", "finalize"]

    result = graph.invoke(initial_state, config)

    for phase_idx in range(len(phases) - 1):
        # Check progress at this checkpoint
        state_snapshot = graph.get_state(config)
        values = state_snapshot.values

        print(f"  Progress Check (after {phases[phase_idx]}):")
        print(f"    Phase: {values.get('current_phase', 'N/A')}")
        print(f"    Extract: {values.get('extract_progress', 0)}%")
        print(f"    Transform: {values.get('transform_progress', 0)}%")
        print(f"    Load: {values.get('load_progress', 0)}%")
        print(f"    Next: {state_snapshot.next}")
        print()

        # Resume to next phase
        if state_snapshot.next:
            result = graph.invoke(None, config)

    print("  Final state:")
    print(f"    Complete: {result.get('is_complete', False)}")
    print(f"    Records migrated: {result.get('records_loaded', 0)}/{result.get('total_records', 0)}")
    print()
    print("  In production, a monitoring service would poll get_state()")
    print("  periodically to display progress in a dashboard UI.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all long-running workflow demonstrations."""
    print("=" * 60)
    print("  LangGraph: Long Running Workflows")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates long-running workflow patterns.")
    print("No LLM calls are needed - we focus on persistence and progress.")
    print()
    print("Topics covered:")
    print("  1. Multi-phase pipelines with progress tracking")
    print("  2. Pause and resume after system interruptions")
    print("  3. Timeout detection and retry with exponential backoff")
    print("  4. Progress monitoring via checkpoint polling")

    # Part 1: Data migration pipeline
    demonstrate_migration_pipeline()

    # Part 2: Pause and resume
    demonstrate_pause_and_resume()

    # Part 3: Timeout and retry handling
    demonstrate_timeout_retry()

    # Part 4: Progress monitoring
    demonstrate_progress_monitoring()

    print("\n" + "=" * 60)
    print("  Exercise 22 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Checkpointing preserves progress across system restarts")
    print("  2. Each phase reports progress percentage through state updates")
    print("  3. interrupt_after enables pausing between phases for monitoring")
    print("  4. Timeout detection compares operation duration to threshold")
    print("  5. Retry logic uses conditional edges to loop back on failure")
    print("  6. Exponential backoff prevents overwhelming failing systems")
    print("  7. get_state() enables external progress monitoring without interruption")
    print("  8. Long-running workflows should be designed as resumable phases")
    print("  9. State tracks both progress metrics and error history")
    print()


if __name__ == "__main__":
    main()
