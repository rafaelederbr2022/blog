"""
Module 3 - Exercise 21: Approval Workflow
==========================================
Build a multi-step approval workflow using LangGraph's interrupt mechanism.
This exercise demonstrates a document publishing pipeline that requires
multiple approval gates before content can be published.

Concepts covered:
- Multi-step approval workflows with multiple gates
- Handling revision loops (rejected -> revise -> re-submit)
- Using MemorySaver + interrupt_before for approval points
- Multiple approval stages (content review, legal review, final sign-off)
- Full cycle: submit, pause for approval, approve/reject, continue/revise

Example: A document publishing pipeline with draft -> review -> approve/reject
-> publish/revise flow.
"""

import sys
sys.path.append('..')
from config import load_config

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class DocumentState(TypedDict):
    """State for the document publishing approval workflow.

    Fields:
        title: Document title
        content: Document content/body
        author: Name of the document author
        version: Current version number (increments on revision)
        stage: Current stage in the pipeline
        content_review_status: Result of content review (approved/rejected/pending)
        content_review_notes: Notes from content reviewer
        legal_review_status: Result of legal review (approved/rejected/pending)
        legal_review_notes: Notes from legal reviewer
        final_signoff_status: Result of final sign-off (approved/rejected/pending)
        final_signoff_notes: Notes from final approver
        revision_history: List of revision notes from rejections
        is_published: Whether the document has been published
        steps_log: Log of all steps executed
    """
    title: str
    content: str
    author: str
    version: int
    stage: str
    content_review_status: str
    content_review_notes: str
    legal_review_status: str
    legal_review_notes: str
    final_signoff_status: str
    final_signoff_notes: str
    revision_history: list[str]
    is_published: bool
    steps_log: list[str]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def submit_draft(state: DocumentState) -> dict:
    """Submit or re-submit a document draft for review."""
    version = state.get("version", 1)
    title = state["title"]
    author = state["author"]

    if version == 1:
        log_entry = f"[submit_draft] New document submitted: '{title}' by {author}"
    else:
        log_entry = f"[submit_draft] Revised document (v{version}) re-submitted: '{title}' by {author}"

    print(f"  -> {log_entry}")

    return {
        "stage": "submitted",
        "steps_log": state.get("steps_log", []) + [log_entry],
    }


def content_review(state: DocumentState) -> dict:
    """Content review gate - checks quality and accuracy of the document.

    The graph pauses here (via interrupt_before) so a human content
    reviewer can approve or reject the document.
    """
    title = state["title"]
    version = state["version"]

    log_entry = f"[content_review] Document '{title}' (v{version}) awaiting content review..."
    print(f"  -> {log_entry}")
    print(f"     Content preview: {state['content'][:80]}...")
    print(f"     Reviewer should check: accuracy, clarity, completeness")

    return {
        "stage": "content_review",
        "steps_log": state["steps_log"] + [log_entry],
    }


def legal_review(state: DocumentState) -> dict:
    """Legal review gate - checks compliance and legal issues.

    The graph pauses here (via interrupt_before) so a legal reviewer
    can approve or reject the document.
    """
    title = state["title"]

    log_entry = f"[legal_review] Document '{title}' passed content review. Awaiting legal review..."
    print(f"  -> {log_entry}")
    print(f"     Legal reviewer should check: compliance, IP issues, disclaimers")

    return {
        "stage": "legal_review",
        "steps_log": state["steps_log"] + [log_entry],
    }


def final_signoff(state: DocumentState) -> dict:
    """Final sign-off gate - executive approval before publishing.

    The graph pauses here (via interrupt_before) so a senior approver
    can give the final go-ahead.
    """
    title = state["title"]

    log_entry = f"[final_signoff] Document '{title}' passed legal review. Awaiting final sign-off..."
    print(f"  -> {log_entry}")
    print(f"     Final approver should confirm: ready for public release")

    return {
        "stage": "final_signoff",
        "steps_log": state["steps_log"] + [log_entry],
    }


def publish_document(state: DocumentState) -> dict:
    """Publish the approved document."""
    title = state["title"]
    version = state["version"]

    log_entry = f"[publish] Document '{title}' (v{version}) PUBLISHED successfully!"
    print(f"  -> {log_entry}")

    return {
        "stage": "published",
        "is_published": True,
        "steps_log": state["steps_log"] + [log_entry],
    }


def revise_document(state: DocumentState) -> dict:
    """Revise the document based on rejection feedback.

    Collects rejection notes and increments the version number.
    After revision, the document goes back to the submission stage.
    """
    version = state["version"]
    new_version = version + 1

    # Collect the rejection reason from whichever stage rejected
    rejection_reason = ""
    if state["content_review_status"] == "rejected":
        rejection_reason = f"Content review (v{version}): {state['content_review_notes']}"
    elif state["legal_review_status"] == "rejected":
        rejection_reason = f"Legal review (v{version}): {state['legal_review_notes']}"
    elif state["final_signoff_status"] == "rejected":
        rejection_reason = f"Final sign-off (v{version}): {state['final_signoff_notes']}"

    log_entry = f"[revise] Revising document to v{new_version}. Reason: {rejection_reason}"
    print(f"  -> {log_entry}")

    # Simulate revision by appending a note to content
    revised_content = state["content"] + f" [Revised in v{new_version}]"

    return {
        "version": new_version,
        "content": revised_content,
        "stage": "revision",
        "content_review_status": "pending",
        "content_review_notes": "",
        "legal_review_status": "pending",
        "legal_review_notes": "",
        "final_signoff_status": "pending",
        "final_signoff_notes": "",
        "revision_history": state.get("revision_history", []) + [rejection_reason],
        "steps_log": state["steps_log"] + [log_entry],
    }


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_after_content_review(state: DocumentState) -> str:
    """Route based on content review decision."""
    if state["content_review_status"] == "approved":
        return "legal_review"
    elif state["content_review_status"] == "rejected":
        return "revise"
    # Default: stay pending (shouldn't happen in normal flow)
    return "legal_review"


def route_after_legal_review(state: DocumentState) -> str:
    """Route based on legal review decision."""
    if state["legal_review_status"] == "approved":
        return "final_signoff"
    elif state["legal_review_status"] == "rejected":
        return "revise"
    return "final_signoff"


def route_after_final_signoff(state: DocumentState) -> str:
    """Route based on final sign-off decision."""
    if state["final_signoff_status"] == "approved":
        return "publish"
    elif state["final_signoff_status"] == "rejected":
        return "revise"
    return "publish"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_approval_workflow() -> tuple:
    """Build the document approval workflow graph.

    Returns:
        Tuple of (compiled graph, checkpointer)
    """
    checkpointer = MemorySaver()

    builder = StateGraph(DocumentState)

    # Add nodes
    builder.add_node("submit_draft", submit_draft)
    builder.add_node("content_review", content_review)
    builder.add_node("legal_review", legal_review)
    builder.add_node("final_signoff", final_signoff)
    builder.add_node("publish", publish_document)
    builder.add_node("revise", revise_document)

    # Add edges
    builder.add_edge(START, "submit_draft")
    builder.add_edge("submit_draft", "content_review")

    # After content review: approve -> legal, reject -> revise
    builder.add_conditional_edges(
        "content_review",
        route_after_content_review,
        {"legal_review": "legal_review", "revise": "revise"},
    )

    # After legal review: approve -> final, reject -> revise
    builder.add_conditional_edges(
        "legal_review",
        route_after_legal_review,
        {"final_signoff": "final_signoff", "revise": "revise"},
    )

    # After final sign-off: approve -> publish, reject -> revise
    builder.add_conditional_edges(
        "final_signoff",
        route_after_final_signoff,
        {"publish": "publish", "revise": "revise"},
    )

    # Revision loops back to submit
    builder.add_edge("revise", "submit_draft")

    # Publish ends the workflow
    builder.add_edge("publish", END)

    # Compile with interrupt_before on all review nodes
    # This pauses the graph before each review so a human can provide input
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["content_review", "legal_review", "final_signoff"],
    )

    return graph, checkpointer


# ---------------------------------------------------------------------------
# Part 1: Full Approval Flow (All Stages Approved)
# ---------------------------------------------------------------------------

def demonstrate_full_approval():
    """Demonstrate the happy path: document approved at all stages."""
    print("\n" + "=" * 60)
    print("PART 1: Full Approval Flow (Happy Path)")
    print("=" * 60)
    print()
    print("A document goes through all three approval gates successfully:")
    print("  submit -> content_review -> legal_review -> final_signoff -> publish")
    print()
    print("The graph pauses before each review node (interrupt_before),")
    print("allowing a human to set the approval status before resuming.")
    print()

    graph, checkpointer = build_approval_workflow()

    config = {"configurable": {"thread_id": "doc-001"}}

    initial_state = {
        "title": "Q4 Product Roadmap",
        "content": "This document outlines our product strategy for Q4 2024, including new features, timeline, and resource allocation.",
        "author": "Sarah Chen",
        "version": 1,
        "stage": "draft",
        "content_review_status": "pending",
        "content_review_notes": "",
        "legal_review_status": "pending",
        "legal_review_notes": "",
        "final_signoff_status": "pending",
        "final_signoff_notes": "",
        "revision_history": [],
        "is_published": False,
        "steps_log": [],
    }

    # Step 1: Submit the document (runs submit_draft, pauses before content_review)
    print("  Step 1: Submitting document...")
    print("-" * 40)
    result = graph.invoke(initial_state, config)
    print()
    print(f"  Graph paused before: {graph.get_state(config).next}")
    print()

    # Step 2: Content reviewer approves
    print("  Step 2: Content reviewer APPROVES the document")
    print("-" * 40)
    graph.update_state(
        config,
        {
            "content_review_status": "approved",
            "content_review_notes": "Well-written, clear objectives. Approved.",
        },
    )
    result = graph.invoke(None, config)
    print()
    print(f"  Graph paused before: {graph.get_state(config).next}")
    print()

    # Step 3: Legal reviewer approves
    print("  Step 3: Legal reviewer APPROVES the document")
    print("-" * 40)
    graph.update_state(
        config,
        {
            "legal_review_status": "approved",
            "legal_review_notes": "No compliance issues found. Cleared.",
        },
    )
    result = graph.invoke(None, config)
    print()
    print(f"  Graph paused before: {graph.get_state(config).next}")
    print()

    # Step 4: Final sign-off approves
    print("  Step 4: Executive gives FINAL SIGN-OFF")
    print("-" * 40)
    graph.update_state(
        config,
        {
            "final_signoff_status": "approved",
            "final_signoff_notes": "Looks great. Publish it.",
        },
    )
    result = graph.invoke(None, config)
    print()

    print("  Document published successfully!")
    print(f"  Title: {result['title']}")
    print(f"  Version: {result['version']}")
    print(f"  Published: {result['is_published']}")
    print()


# ---------------------------------------------------------------------------
# Part 2: Rejection and Revision Loop
# ---------------------------------------------------------------------------

def demonstrate_rejection_loop():
    """Demonstrate rejection at content review, revision, and re-approval."""
    print("\n" + "=" * 60)
    print("PART 2: Rejection and Revision Loop")
    print("=" * 60)
    print()
    print("A document is rejected at content review, revised, and re-submitted.")
    print("This demonstrates the revision loop:")
    print("  submit -> content_review -> REJECT -> revise -> submit -> content_review -> APPROVE -> ...")
    print()

    graph, checkpointer = build_approval_workflow()

    config = {"configurable": {"thread_id": "doc-002"}}

    initial_state = {
        "title": "API Security Guidelines",
        "content": "Draft security guidelines for our public API endpoints.",
        "author": "Mike Torres",
        "version": 1,
        "stage": "draft",
        "content_review_status": "pending",
        "content_review_notes": "",
        "legal_review_status": "pending",
        "legal_review_notes": "",
        "final_signoff_status": "pending",
        "final_signoff_notes": "",
        "revision_history": [],
        "is_published": False,
        "steps_log": [],
    }

    # Step 1: Submit
    print("  Step 1: Submitting document (v1)...")
    print("-" * 40)
    result = graph.invoke(initial_state, config)
    print()

    # Step 2: Content reviewer REJECTS
    print("  Step 2: Content reviewer REJECTS the document")
    print("-" * 40)
    print("  Reason: Missing authentication examples and rate limiting section")
    print()
    graph.update_state(
        config,
        {
            "content_review_status": "rejected",
            "content_review_notes": "Missing authentication examples and rate limiting section. Please add.",
        },
    )
    result = graph.invoke(None, config)
    print()
    print(f"  Document sent back for revision. New version: {result['version']}")
    print(f"  Revision history: {result['revision_history']}")
    print()

    # The graph loops back: revise -> submit_draft -> pauses before content_review
    print(f"  Graph paused before: {graph.get_state(config).next}")
    print()

    # Step 3: Content reviewer APPROVES the revised version
    print("  Step 3: Content reviewer APPROVES revised document (v2)")
    print("-" * 40)
    graph.update_state(
        config,
        {
            "content_review_status": "approved",
            "content_review_notes": "Revision addresses all concerns. Approved.",
        },
    )
    result = graph.invoke(None, config)
    print()
    print(f"  Graph paused before: {graph.get_state(config).next}")
    print()

    # Step 4: Legal approves
    print("  Step 4: Legal reviewer APPROVES")
    print("-" * 40)
    graph.update_state(
        config,
        {
            "legal_review_status": "approved",
            "legal_review_notes": "Compliant with data protection regulations.",
        },
    )
    result = graph.invoke(None, config)
    print()

    # Step 5: Final sign-off
    print("  Step 5: Executive gives FINAL SIGN-OFF")
    print("-" * 40)
    graph.update_state(
        config,
        {
            "final_signoff_status": "approved",
            "final_signoff_notes": "Critical document. Publish immediately.",
        },
    )
    result = graph.invoke(None, config)
    print()

    print("  Document published after revision!")
    print(f"  Final version: v{result['version']}")
    print(f"  Revision history: {result['revision_history']}")
    print(f"  Published: {result['is_published']}")
    print()


# ---------------------------------------------------------------------------
# Part 3: Multiple Rejections at Different Stages
# ---------------------------------------------------------------------------

def demonstrate_multi_stage_rejection():
    """Demonstrate rejections at different stages of the pipeline."""
    print("\n" + "=" * 60)
    print("PART 3: Multiple Rejections at Different Stages")
    print("=" * 60)
    print()
    print("A document passes content review but is rejected by legal,")
    print("then after revision passes all stages.")
    print()

    graph, checkpointer = build_approval_workflow()

    config = {"configurable": {"thread_id": "doc-003"}}

    initial_state = {
        "title": "Partner Integration Agreement",
        "content": "Terms and conditions for third-party API integrations with our platform.",
        "author": "Lisa Park",
        "version": 1,
        "stage": "draft",
        "content_review_status": "pending",
        "content_review_notes": "",
        "legal_review_status": "pending",
        "legal_review_notes": "",
        "final_signoff_status": "pending",
        "final_signoff_notes": "",
        "revision_history": [],
        "is_published": False,
        "steps_log": [],
    }

    # Submit
    print("  Submitting document (v1)...")
    result = graph.invoke(initial_state, config)
    print()

    # Content review: APPROVE
    print("  Content review: APPROVED")
    graph.update_state(config, {
        "content_review_status": "approved",
        "content_review_notes": "Content is clear and complete.",
    })
    result = graph.invoke(None, config)
    print()

    # Legal review: REJECT
    print("  Legal review: REJECTED")
    print("  Reason: Missing data processing addendum (DPA) clause")
    graph.update_state(config, {
        "legal_review_status": "rejected",
        "legal_review_notes": "Missing DPA clause required by GDPR. Must include data processing terms.",
    })
    result = graph.invoke(None, config)
    print()
    print(f"  Sent back for revision. Now at v{result['version']}")
    print()

    # After revision, document goes back through ALL review stages
    # Content review again: APPROVE
    print("  Content review (v2): APPROVED")
    graph.update_state(config, {
        "content_review_status": "approved",
        "content_review_notes": "DPA clause added correctly.",
    })
    result = graph.invoke(None, config)
    print()

    # Legal review again: APPROVE
    print("  Legal review (v2): APPROVED")
    graph.update_state(config, {
        "legal_review_status": "approved",
        "legal_review_notes": "DPA clause meets GDPR requirements. Cleared.",
    })
    result = graph.invoke(None, config)
    print()

    # Final sign-off: APPROVE
    print("  Final sign-off: APPROVED")
    graph.update_state(config, {
        "final_signoff_status": "approved",
        "final_signoff_notes": "Approved for partner distribution.",
    })
    result = graph.invoke(None, config)
    print()

    print("  Document published after legal revision!")
    print(f"  Final version: v{result['version']}")
    print(f"  Revision history: {result['revision_history']}")
    print(f"  Total steps: {len(result['steps_log'])}")
    print()


# ---------------------------------------------------------------------------
# Part 4: Inspecting Workflow State During Approval
# ---------------------------------------------------------------------------

def demonstrate_state_inspection():
    """Show how to inspect the workflow state at each approval point."""
    print("\n" + "=" * 60)
    print("PART 4: Inspecting Workflow State During Approval")
    print("=" * 60)
    print()
    print("When the graph pauses at an approval gate, you can inspect")
    print("the full state to make an informed decision.")
    print()

    graph, checkpointer = build_approval_workflow()

    config = {"configurable": {"thread_id": "doc-004"}}

    initial_state = {
        "title": "Annual Budget Proposal",
        "content": "Proposed budget allocation for FY2025 across all departments.",
        "author": "James Wright",
        "version": 1,
        "stage": "draft",
        "content_review_status": "pending",
        "content_review_notes": "",
        "legal_review_status": "pending",
        "legal_review_notes": "",
        "final_signoff_status": "pending",
        "final_signoff_notes": "",
        "revision_history": [],
        "is_published": False,
        "steps_log": [],
    }

    # Submit and pause
    result = graph.invoke(initial_state, config)
    print()

    # Inspect state at the pause point
    print("  Inspecting state at content_review gate:")
    print("-" * 40)
    current_state = graph.get_state(config)

    print(f"    Next node(s): {current_state.next}")
    print(f"    Document: '{current_state.values['title']}'")
    print(f"    Author: {current_state.values['author']}")
    print(f"    Version: {current_state.values['version']}")
    print(f"    Stage: {current_state.values['stage']}")
    print(f"    Content preview: {current_state.values['content'][:60]}...")
    print()
    print("  This information helps the reviewer make an informed decision.")
    print("  In a real application, this state would be displayed in a UI.")
    print()

    # Check history
    history = list(graph.get_state_history(config))
    print(f"  Checkpoint history entries: {len(history)}")
    for i, snapshot in enumerate(history[:5]):
        stage = snapshot.values.get("stage", "N/A")
        print(f"    [{i}] stage='{stage}', next={snapshot.next}")

    print()
    print("  The checkpoint history allows auditing of the full approval process.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all approval workflow demonstrations."""
    print("=" * 60)
    print("  LangGraph: Approval Workflow")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates multi-step approval workflows.")
    print("No LLM calls are needed - we focus on approval mechanics.")
    print()
    print("Workflow structure:")
    print("  submit -> content_review -> legal_review -> final_signoff -> publish")
    print("                |                  |                  |")
    print("                v                  v                  v")
    print("             [reject]           [reject]          [reject]")
    print("                |                  |                  |")
    print("                +---------> revise <-----------------+")
    print("                              |")
    print("                              v")
    print("                           submit (loop back)")

    # Part 1: Full approval (happy path)
    demonstrate_full_approval()

    # Part 2: Rejection and revision loop
    demonstrate_rejection_loop()

    # Part 3: Multiple rejections at different stages
    demonstrate_multi_stage_rejection()

    # Part 4: State inspection during approval
    demonstrate_state_inspection()

    print("\n" + "=" * 60)
    print("  Exercise 21 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. interrupt_before on review nodes creates approval gates")
    print("  2. graph.update_state() sets the approval decision before resuming")
    print("  3. Conditional edges route to 'revise' on rejection or next stage on approval")
    print("  4. Revision loops back to submit, restarting the review cycle")
    print("  5. Each revision increments the version and records history")
    print("  6. The full audit trail is preserved in checkpoint history")
    print("  7. State inspection at pause points enables informed decisions")
    print("  8. Multiple approval gates can be chained sequentially")
    print()


if __name__ == "__main__":
    main()
