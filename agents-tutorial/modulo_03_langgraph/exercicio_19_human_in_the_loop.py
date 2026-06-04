"""
Module 3 - Exercise 19: Human-in-the-Loop
==========================================
Learn how to pause graph execution for human intervention using LangGraph's
interrupt mechanism. Human-in-the-loop patterns are essential for workflows
that require human approval, review, or input before proceeding.

Concepts covered:
- Using interrupt_before and interrupt_after to pause graph execution
- Pausing the graph at a specific node for human approval
- Using MemorySaver checkpointer to enable interrupts
- Simulating human input (approve/reject) and resuming execution
- Updating state after human review and resuming the graph

Example: An expense approval workflow where amounts > $500 require human approval.
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

class ExpenseState(TypedDict):
    """State for the expense approval workflow.

    Fields:
        employee: Name of the employee submitting the expense
        amount: Dollar amount of the expense
        description: Description of the expense
        category: Expense category (travel, equipment, meals, etc.)
        requires_approval: Whether the expense needs human approval
        approval_status: Status after human review (approved/rejected/None)
        approval_notes: Notes from the approver
        processing_result: Final processing result message
        steps_log: Log of steps executed
    """
    employee: str
    amount: float
    description: str
    category: str
    requires_approval: bool
    approval_status: Optional[str]
    approval_notes: str
    processing_result: str
    steps_log: list[str]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def submit_expense(state: ExpenseState) -> dict:
    """First node: receives and validates the expense submission."""
    employee = state["employee"]
    amount = state["amount"]
    description = state["description"]

    log_entry = f"[submit_expense] {employee} submitted ${amount:.2f} for '{description}'"
    print(f"  -> {log_entry}")

    return {
        "steps_log": state.get("steps_log", []) + [log_entry],
    }


def check_threshold(state: ExpenseState) -> dict:
    """Second node: checks if the amount exceeds the approval threshold ($500)."""
    amount = state["amount"]
    threshold = 500.0
    needs_approval = amount > threshold

    if needs_approval:
        log_entry = f"[check_threshold] Amount ${amount:.2f} > ${threshold:.2f} - REQUIRES human approval"
    else:
        log_entry = f"[check_threshold] Amount ${amount:.2f} <= ${threshold:.2f} - Auto-approved"

    print(f"  -> {log_entry}")

    return {
        "requires_approval": needs_approval,
        "steps_log": state["steps_log"] + [log_entry],
    }


def human_review(state: ExpenseState) -> dict:
    """Third node: this is where the graph pauses for human review.

    In a real application, this node would be preceded by an interrupt,
    and the human would update the state before resuming.
    For this demonstration, we simulate the human decision.
    """
    log_entry = "[human_review] Waiting for human approval..."
    print(f"  -> {log_entry}")
    print(f"     Expense: ${state['amount']:.2f} - {state['description']}")
    print(f"     Employee: {state['employee']}")
    print(f"     Category: {state['category']}")

    return {
        "steps_log": state["steps_log"] + [log_entry],
    }


def process_expense(state: ExpenseState) -> dict:
    """Final node: processes the expense based on approval status."""
    if state["requires_approval"]:
        if state["approval_status"] == "approved":
            result = f"Expense APPROVED and processed: ${state['amount']:.2f} for {state['description']}"
            notes = state.get("approval_notes", "")
            if notes:
                result += f" (Notes: {notes})"
        elif state["approval_status"] == "rejected":
            result = f"Expense REJECTED: ${state['amount']:.2f} for {state['description']}"
            notes = state.get("approval_notes", "")
            if notes:
                result += f" (Reason: {notes})"
        else:
            result = f"Expense PENDING: awaiting approval decision"
    else:
        result = f"Expense AUTO-APPROVED and processed: ${state['amount']:.2f} for {state['description']}"
        
    log_entry = f"[process_expense] {result}"
    print(f"  -> {log_entry}")

    return {
        "processing_result": result,
        "steps_log": state["steps_log"] + [log_entry],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------

def route_after_threshold(state: ExpenseState) -> str:
    """Route based on whether human approval is required."""
    if state["requires_approval"]:
        return "human_review"
    else:
        return "process_expense"


# ---------------------------------------------------------------------------
# Part 1: Basic Human-in-the-Loop with interrupt_before
# ---------------------------------------------------------------------------

def demonstrate_interrupt_before():
    """Show how interrupt_before pauses execution BEFORE a node runs."""
    print("\n" + "=" * 60)
    print("PART 1: Human-in-the-Loop with interrupt_before")
    print("=" * 60)
    print()
    print("interrupt_before pauses the graph BEFORE a specified node executes.")
    print("This allows a human to review the state and decide whether to proceed.")
    print()
    print("Use case: Pause before processing an expense that needs approval,")
    print("so a manager can review and approve/reject it.")
    print()

    # Step 1: Create the graph with MemorySaver (required for interrupts)
    print("Step 1: Create graph with MemorySaver checkpointer")
    print("-" * 40)
    print("  MemorySaver is required to enable interrupts.")
    print("  It saves the graph state so execution can be resumed later.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(ExpenseState)
    builder.add_node("submit", submit_expense)
    builder.add_node("check_threshold", check_threshold)
    builder.add_node("human_review", human_review)
    builder.add_node("process", process_expense)

    builder.add_edge(START, "submit")
    builder.add_edge("submit", "check_threshold")
    builder.add_conditional_edges(
        "check_threshold",
        route_after_threshold,
        {"human_review": "human_review", "process_expense": "process"},
    )
    builder.add_edge("human_review", "process")
    builder.add_edge("process", END)

    # Compile with interrupt_before on the "process" node
    # This means: after human_review runs, pause BEFORE process runs
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["process"],
    )

    print("  Graph compiled with interrupt_before=['process']")
    print("  Flow: submit -> check_threshold -> [human_review] -> PAUSE -> process")
    print()

    # Step 2: Run a high-value expense (will trigger approval)
    print("Step 2: Submit a high-value expense ($1,200)")
    print("-" * 40)
    print()

    # thread_id identifies this specific execution session
    config = {"configurable": {"thread_id": "expense-001"}}

    initial_state = {
        "employee": "Alice Johnson",
        "amount": 1200.00,
        "description": "Conference registration and travel",
        "category": "travel",
        "requires_approval": False,
        "approval_status": None,
        "approval_notes": "",
        "processing_result": "",
        "steps_log": [],
    }

    print("  Invoking graph (will pause before 'process' node)...")
    print()

    # First invocation - will pause before "process" node
    result = graph.invoke(initial_state, config)

    print()
    print("  Graph PAUSED! Execution stopped before 'process' node.")
    print(f"  Current state - requires_approval: {result['requires_approval']}")
    print(f"  Current state - approval_status: {result['approval_status']}")
    print()

    # Step 3: Simulate human approval by updating state
    print("Step 3: Human approves the expense (update state and resume)")
    print("-" * 40)
    print()
    print("  Manager reviews: $1,200 for 'Conference registration and travel'")
    print("  Decision: APPROVED")
    print()

    # Update the state with human decision
    graph.update_state(
        config,
        {
            "approval_status": "approved",
            "approval_notes": "Approved - important industry conference",
        },
    )

    print("  State updated with approval. Resuming graph execution...")
    print()

    # Resume execution - pass None as input to continue from checkpoint
    final_result = graph.invoke(None, config)

    print()
    print("  Final result:")
    print(f"    processing_result: {final_result['processing_result']}")
    print()
    print("  Execution log:")
    for step in final_result["steps_log"]:
        print(f"    - {step}")

    return graph, checkpointer


# ---------------------------------------------------------------------------
# Part 2: Demonstrating Rejection Flow
# ---------------------------------------------------------------------------

def demonstrate_rejection_flow():
    """Show the rejection path in the human-in-the-loop workflow."""
    print("\n" + "=" * 60)
    print("PART 2: Human Rejection Flow")
    print("=" * 60)
    print()
    print("Same workflow, but this time the human rejects the expense.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(ExpenseState)
    builder.add_node("submit", submit_expense)
    builder.add_node("check_threshold", check_threshold)
    builder.add_node("human_review", human_review)
    builder.add_node("process", process_expense)

    builder.add_edge(START, "submit")
    builder.add_edge("submit", "check_threshold")
    builder.add_conditional_edges(
        "check_threshold",
        route_after_threshold,
        {"human_review": "human_review", "process_expense": "process"},
    )
    builder.add_edge("human_review", "process")
    builder.add_edge("process", END)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["process"],
    )

    config = {"configurable": {"thread_id": "expense-002"}}

    initial_state = {
        "employee": "Bob Smith",
        "amount": 3500.00,
        "description": "New standing desk and ergonomic chair",
        "category": "equipment",
        "requires_approval": False,
        "approval_status": None,
        "approval_notes": "",
        "processing_result": "",
        "steps_log": [],
    }

    print("  Submitting expense: $3,500 for equipment...")
    print()

    # Run until interrupt
    result = graph.invoke(initial_state, config)

    print()
    print("  Graph PAUSED before processing.")
    print()
    print("  Manager reviews: $3,500 for 'New standing desk and ergonomic chair'")
    print("  Decision: REJECTED (over budget for this quarter)")
    print()

    # Human rejects
    graph.update_state(
        config,
        {
            "approval_status": "rejected",
            "approval_notes": "Over budget for Q4. Resubmit in Q1.",
        },
    )

    # Resume
    final_result = graph.invoke(None, config)

    print()
    print("  Final result:")
    print(f"    processing_result: {final_result['processing_result']}")
    print()


# ---------------------------------------------------------------------------
# Part 3: Auto-Approval (No Interrupt Triggered)
# ---------------------------------------------------------------------------

def demonstrate_auto_approval():
    """Show that low-value expenses bypass the human review entirely."""
    print("\n" + "=" * 60)
    print("PART 3: Auto-Approval (Below Threshold)")
    print("=" * 60)
    print()
    print("Expenses <= $500 are auto-approved and skip human review.")
    print("The interrupt_before on 'process' still exists, but the conditional")
    print("routing sends low-value expenses directly to 'process' without")
    print("going through 'human_review' first.")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(ExpenseState)
    builder.add_node("submit", submit_expense)
    builder.add_node("check_threshold", check_threshold)
    builder.add_node("human_review", human_review)
    builder.add_node("process", process_expense)

    builder.add_edge(START, "submit")
    builder.add_edge("submit", "check_threshold")
    builder.add_conditional_edges(
        "check_threshold",
        route_after_threshold,
        {"human_review": "human_review", "process_expense": "process"},
    )
    builder.add_edge("human_review", "process")
    builder.add_edge("process", END)

    # For auto-approval, we only interrupt when coming from human_review
    # We use interrupt_after on human_review instead
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["human_review"],
    )

    config = {"configurable": {"thread_id": "expense-003"}}

    initial_state = {
        "employee": "Carol Davis",
        "amount": 45.00,
        "description": "Team lunch",
        "category": "meals",
        "requires_approval": False,
        "approval_status": None,
        "approval_notes": "",
        "processing_result": "",
        "steps_log": [],
    }

    print("  Submitting low-value expense: $45 for team lunch...")
    print()

    # This will run to completion without pausing (no human_review node visited)
    result = graph.invoke(initial_state, config)

    print()
    print("  Graph completed WITHOUT pausing (amount below threshold).")
    print(f"  Result: {result['processing_result']}")
    print()
    print("  Key insight: interrupt_after=['human_review'] only triggers if")
    print("  the 'human_review' node actually executes. Since the conditional")
    print("  routing skipped it, no interrupt occurred.")


# ---------------------------------------------------------------------------
# Part 4: interrupt_after Demonstration
# ---------------------------------------------------------------------------

def demonstrate_interrupt_after():
    """Show how interrupt_after pauses execution AFTER a node completes."""
    print("\n" + "=" * 60)
    print("PART 4: interrupt_after vs interrupt_before")
    print("=" * 60)
    print()
    print("interrupt_after pauses AFTER a node completes (but before the next).")
    print("interrupt_before pauses BEFORE a node starts.")
    print()
    print("Comparison:")
    print("  interrupt_before=['process']  -> pause BEFORE process runs")
    print("  interrupt_after=['human_review'] -> pause AFTER human_review runs")
    print()
    print("Both achieve similar results but differ in timing:")
    print("  - interrupt_before: the paused node hasn't executed yet")
    print("  - interrupt_after: the specified node has already executed")
    print()

    checkpointer = MemorySaver()

    builder = StateGraph(ExpenseState)
    builder.add_node("submit", submit_expense)
    builder.add_node("check_threshold", check_threshold)
    builder.add_node("human_review", human_review)
    builder.add_node("process", process_expense)

    builder.add_edge(START, "submit")
    builder.add_edge("submit", "check_threshold")
    builder.add_conditional_edges(
        "check_threshold",
        route_after_threshold,
        {"human_review": "human_review", "process_expense": "process"},
    )
    builder.add_edge("human_review", "process")
    builder.add_edge("process", END)

    # Using interrupt_after on human_review
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["human_review"],
    )

    config = {"configurable": {"thread_id": "expense-004"}}

    initial_state = {
        "employee": "Dave Wilson",
        "amount": 800.00,
        "description": "Software license renewal",
        "category": "software",
        "requires_approval": False,
        "approval_status": None,
        "approval_notes": "",
        "processing_result": "",
        "steps_log": [],
    }

    print("  Using interrupt_after=['human_review']")
    print("  Submitting: $800 for software license...")
    print()

    result = graph.invoke(initial_state, config)

    print()
    print("  Graph PAUSED after 'human_review' node completed.")
    print("  The human_review node has already run (see log above).")
    print("  Now waiting for human to update approval_status before resuming.")
    print()

    # Human approves
    graph.update_state(
        config,
        {
            "approval_status": "approved",
            "approval_notes": "Essential tool - approved",
        },
    )

    final_result = graph.invoke(None, config)

    print()
    print(f"  Final: {final_result['processing_result']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all human-in-the-loop demonstrations."""
    print("=" * 60)
    print("  LangGraph: Human-in-the-Loop")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates human-in-the-loop patterns in LangGraph.")
    print("No LLM calls are needed - we focus on interrupt and resume mechanics.")

    # Part 1: Basic interrupt_before with approval
    demonstrate_interrupt_before()

    # Part 2: Rejection flow
    demonstrate_rejection_flow()

    # Part 3: Auto-approval (no interrupt triggered)
    demonstrate_auto_approval()

    # Part 4: interrupt_after demonstration
    demonstrate_interrupt_after()

    print("\n" + "=" * 60)
    print("  Exercise 19 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. MemorySaver checkpointer is REQUIRED to enable interrupts")
    print("  2. interrupt_before pauses BEFORE a node executes")
    print("  3. interrupt_after pauses AFTER a node completes")
    print("  4. Use thread_id in config to identify execution sessions")
    print("  5. graph.update_state() lets you modify state during a pause")
    print("  6. graph.invoke(None, config) resumes from the last checkpoint")
    print("  7. Conditional routing can skip interrupt points entirely")
    print("  8. Human-in-the-loop is essential for approval workflows")
    print()


if __name__ == "__main__":
    main()
