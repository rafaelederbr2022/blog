"""
Module 3 - Exercise 17: Conditional Routing in LangGraph
=========================================================
Learn how to use add_conditional_edges() to route graph execution
based on state conditions. Conditional routing allows a graph to take
different paths depending on the current state, enabling dynamic
decision-making without an LLM.

Concepts covered:
- Using add_conditional_edges() to define branching logic
- Writing routing functions that inspect state and return next node names
- Multiple conditional branches (more than two paths)
- Combining conditional and fixed edges in the same graph
- Real-world pattern: support ticket routing by category

Example: A support ticket router that classifies tickets and routes them
to different handler nodes (billing, technical, general) based on the
ticket's category field in the state.
"""

import sys
sys.path.append('..')
from config import load_config

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END, START


# ---------------------------------------------------------------------------
# Part 1: Basic Conditional Routing
# ---------------------------------------------------------------------------

class TicketState(TypedDict):
    """State for the support ticket routing system.

    The 'category' field determines which handler processes the ticket.
    """
    ticket_id: str
    customer_name: str
    description: str
    category: str
    priority: str
    resolution: str
    routing_log: Annotated[list[str], operator.add]


def classify_ticket(state: TicketState) -> dict:
    """Classify the ticket based on keywords in the description.

    This node inspects the ticket description and assigns a category.
    In a real system, this could use an LLM or ML model for classification.
    """
    description = state["description"].lower()

    # Simple keyword-based classification
    if any(word in description for word in ["bill", "charge", "payment", "invoice", "refund"]):
        category = "billing"
    elif any(word in description for word in ["error", "bug", "crash", "slow", "not working"]):
        category = "technical"
    elif any(word in description for word in ["urgent", "critical", "down", "outage"]):
        category = "technical"
    else:
        category = "general"

    print(f"  [classify_ticket] Ticket #{state['ticket_id']}: '{state['description'][:50]}...'")
    print(f"  [classify_ticket] Assigned category: '{category}'")

    return {
        "category": category,
        "routing_log": [f"Classified as '{category}'"],
    }


def route_by_category(state: TicketState) -> str:
    """Routing function: inspects state and returns the next node name.

    This is the key function used with add_conditional_edges().
    It must return a string that matches one of the defined node names
    (or END to terminate the graph).
    """
    category = state["category"]
    print(f"  [router] Routing ticket to '{category}' handler...")
    return category


def handle_billing(state: TicketState) -> dict:
    """Handle billing-related tickets."""
    resolution = (
        f"Billing team reviewing ticket #{state['ticket_id']}. "
        f"Customer '{state['customer_name']}' will receive a billing adjustment "
        f"within 2-3 business days."
    )
    print(f"  [handle_billing] Processing billing issue for {state['customer_name']}")
    print(f"  [handle_billing] Resolution: {resolution}")
    return {
        "resolution": resolution,
        "routing_log": ["Routed to billing handler", "Billing resolution applied"],
    }


def handle_technical(state: TicketState) -> dict:
    """Handle technical support tickets."""
    resolution = (
        f"Technical team investigating ticket #{state['ticket_id']}. "
        f"Issue: '{state['description'][:40]}...'. "
        f"An engineer has been assigned and will respond within 1 hour."
    )
    print(f"  [handle_technical] Processing technical issue for {state['customer_name']}")
    print(f"  [handle_technical] Resolution: {resolution}")
    return {
        "resolution": resolution,
        "routing_log": ["Routed to technical handler", "Technical resolution applied"],
    }


def handle_general(state: TicketState) -> dict:
    """Handle general inquiry tickets."""
    resolution = (
        f"General support handling ticket #{state['ticket_id']}. "
        f"Customer '{state['customer_name']}' inquiry has been logged. "
        f"A support agent will follow up via email within 24 hours."
    )
    print(f"  [handle_general] Processing general inquiry for {state['customer_name']}")
    print(f"  [handle_general] Resolution: {resolution}")
    return {
        "resolution": resolution,
        "routing_log": ["Routed to general handler", "General resolution applied"],
    }


def demonstrate_basic_conditional_routing():
    """Show basic conditional routing with add_conditional_edges()."""
    print("\n" + "=" * 60)
    print("PART 1: Basic Conditional Routing - Support Ticket Router")
    print("=" * 60)
    print()
    print("Conditional routing uses add_conditional_edges() to choose the")
    print("next node based on a routing function that inspects the state.")
    print()
    print("Pattern:")
    print("  graph.add_conditional_edges(")
    print("      source_node,        # Node AFTER which routing happens")
    print("      routing_function,   # Function that returns next node name")
    print("      {                   # Mapping of return values to node names")
    print("          'value1': 'node_a',")
    print("          'value2': 'node_b',")
    print("      }")
    print("  )")
    print()

    # Build the graph
    builder = StateGraph(TicketState)

    # Add nodes
    builder.add_node("classify", classify_ticket)
    builder.add_node("billing", handle_billing)
    builder.add_node("technical", handle_technical)
    builder.add_node("general", handle_general)

    # Fixed edge: START -> classify
    builder.add_edge(START, "classify")

    # Conditional edge: classify -> (billing | technical | general)
    builder.add_conditional_edges(
        "classify",          # Source node
        route_by_category,   # Routing function
        {                    # Mapping: routing function return value -> node name
            "billing": "billing",
            "technical": "technical",
            "general": "general",
        }
    )

    # All handlers lead to END
    builder.add_edge("billing", END)
    builder.add_edge("technical", END)
    builder.add_edge("general", END)

    graph = builder.compile()

    # Test with different ticket types
    tickets = [
        {
            "ticket_id": "T-001",
            "customer_name": "Alice Johnson",
            "description": "I was charged twice for my subscription payment last month",
            "category": "",
            "priority": "medium",
            "resolution": "",
            "routing_log": [],
        },
        {
            "ticket_id": "T-002",
            "customer_name": "Bob Smith",
            "description": "The application crashes with an error when I try to export data",
            "category": "",
            "priority": "high",
            "resolution": "",
            "routing_log": [],
        },
        {
            "ticket_id": "T-003",
            "customer_name": "Carol Davis",
            "description": "How do I change my account display name and profile picture?",
            "category": "",
            "priority": "low",
            "resolution": "",
            "routing_log": [],
        },
    ]

    for ticket in tickets:
        print(f"\n{'─' * 50}")
        print(f"  Processing Ticket #{ticket['ticket_id']} from {ticket['customer_name']}")
        print(f"  Description: \"{ticket['description']}\"")
        print(f"{'─' * 50}")
        print()

        result = graph.invoke(ticket)

        print()
        print(f"  Result:")
        print(f"    Category: {result['category']}")
        print(f"    Resolution: {result['resolution'][:80]}...")
        print(f"    Routing log: {result['routing_log']}")


# ---------------------------------------------------------------------------
# Part 2: Multi-Level Conditional Routing
# ---------------------------------------------------------------------------

class OrderState(TypedDict):
    """State for order processing with priority-based routing."""
    order_id: str
    amount: float
    customer_tier: str
    is_international: bool
    processing_result: str
    steps: Annotated[list[str], operator.add]


def analyze_order(state: OrderState) -> dict:
    """Analyze the order and determine processing path."""
    print(f"  [analyze_order] Order #{state['order_id']}: ${state['amount']:.2f}")
    print(f"  [analyze_order] Customer tier: {state['customer_tier']}, International: {state['is_international']}")
    return {
        "steps": [f"Order #{state['order_id']} analyzed"],
    }


def route_by_amount_and_tier(state: OrderState) -> str:
    """Route based on multiple conditions: amount and customer tier.

    Demonstrates routing logic that considers multiple state fields.
    """
    amount = state["amount"]
    tier = state["customer_tier"]
    is_international = state["is_international"]

    if amount > 1000 or tier == "vip":
        route = "premium_processing"
    elif is_international:
        route = "international_processing"
    else:
        route = "standard_processing"

    print(f"  [router] Route decision: '{route}' (amount=${amount:.2f}, tier={tier}, intl={is_international})")
    return route


def premium_processing(state: OrderState) -> dict:
    """Process high-value or VIP orders with priority handling."""
    result = f"PREMIUM: Order #{state['order_id']} fast-tracked. Dedicated account manager assigned."
    print(f"  [premium_processing] {result}")
    return {
        "processing_result": result,
        "steps": ["Routed to premium", "Priority handling applied", "Account manager notified"],
    }


def international_processing(state: OrderState) -> dict:
    """Process international orders with customs and shipping checks."""
    result = f"INTERNATIONAL: Order #{state['order_id']} customs documentation prepared. Extended shipping timeline."
    print(f"  [international_processing] {result}")
    return {
        "processing_result": result,
        "steps": ["Routed to international", "Customs check initiated", "International shipping arranged"],
    }


def standard_processing(state: OrderState) -> dict:
    """Process standard domestic orders."""
    result = f"STANDARD: Order #{state['order_id']} queued for standard processing. Ships in 3-5 days."
    print(f"  [standard_processing] {result}")
    return {
        "processing_result": result,
        "steps": ["Routed to standard", "Standard shipping scheduled"],
    }


def demonstrate_multi_condition_routing():
    """Show routing based on multiple state conditions."""
    print("\n" + "=" * 60)
    print("PART 2: Multi-Condition Routing - Order Processing")
    print("=" * 60)
    print()
    print("The routing function can inspect MULTIPLE state fields to make")
    print("its decision. This enables complex branching logic.")
    print()
    print("In this example, orders are routed based on:")
    print("  - Order amount (> $1000 -> premium)")
    print("  - Customer tier (VIP -> premium)")
    print("  - International flag (True -> international)")
    print("  - Otherwise -> standard")
    print()

    # Build the graph
    builder = StateGraph(OrderState)

    builder.add_node("analyze", analyze_order)
    builder.add_node("premium_processing", premium_processing)
    builder.add_node("international_processing", international_processing)
    builder.add_node("standard_processing", standard_processing)

    builder.add_edge(START, "analyze")

    # Conditional routing based on multiple conditions
    builder.add_conditional_edges(
        "analyze",
        route_by_amount_and_tier,
        {
            "premium_processing": "premium_processing",
            "international_processing": "international_processing",
            "standard_processing": "standard_processing",
        }
    )

    builder.add_edge("premium_processing", END)
    builder.add_edge("international_processing", END)
    builder.add_edge("standard_processing", END)

    graph = builder.compile()

    # Test with different orders
    orders = [
        {
            "order_id": "ORD-100",
            "amount": 2500.00,
            "customer_tier": "standard",
            "is_international": False,
            "processing_result": "",
            "steps": [],
        },
        {
            "order_id": "ORD-101",
            "amount": 50.00,
            "customer_tier": "vip",
            "is_international": False,
            "processing_result": "",
            "steps": [],
        },
        {
            "order_id": "ORD-102",
            "amount": 200.00,
            "customer_tier": "standard",
            "is_international": True,
            "processing_result": "",
            "steps": [],
        },
        {
            "order_id": "ORD-103",
            "amount": 75.00,
            "customer_tier": "basic",
            "is_international": False,
            "processing_result": "",
            "steps": [],
        },
    ]

    for order in orders:
        print(f"\n{'─' * 50}")
        print(f"  Order #{order['order_id']}: ${order['amount']:.2f} | Tier: {order['customer_tier']} | International: {order['is_international']}")
        print(f"{'─' * 50}")

        result = graph.invoke(order)

        print(f"  -> Result: {result['processing_result']}")
        print(f"  -> Steps: {result['steps']}")


# ---------------------------------------------------------------------------
# Part 3: Conditional Routing with END as a Possible Target
# ---------------------------------------------------------------------------

class ValidationState(TypedDict):
    """State for a validation pipeline that can short-circuit."""
    data: str
    is_valid: bool
    error_message: str
    processed_data: str
    validation_steps: Annotated[list[str], operator.add]


def validate_input(state: ValidationState) -> dict:
    """Validate the input data. Sets is_valid flag."""
    data = state["data"]
    errors = []

    if not data:
        errors.append("Data is empty")
    if len(data) < 3:
        errors.append("Data too short (min 3 chars)")
    if any(c in data for c in "!@#$%"):
        errors.append("Data contains invalid characters")

    is_valid = len(errors) == 0
    error_msg = "; ".join(errors) if errors else ""

    print(f"  [validate_input] Data: '{data}' -> Valid: {is_valid}")
    if errors:
        print(f"  [validate_input] Errors: {errors}")

    return {
        "is_valid": is_valid,
        "error_message": error_msg,
        "validation_steps": [f"Validation: {'PASSED' if is_valid else 'FAILED - ' + error_msg}"],
    }


def route_after_validation(state: ValidationState) -> str:
    """Route to processing if valid, or directly to END if invalid.

    This demonstrates using END as a routing target to short-circuit
    the graph when further processing is not needed.
    """
    if state["is_valid"]:
        print(f"  [router] Valid data -> continue to processing")
        return "process"
    else:
        print(f"  [router] Invalid data -> skip to end (short-circuit)")
        return "end"


def process_data(state: ValidationState) -> dict:
    """Process valid data."""
    processed = state["data"].upper().strip()
    print(f"  [process_data] Processed: '{state['data']}' -> '{processed}'")
    return {
        "processed_data": processed,
        "validation_steps": ["Data processed successfully"],
    }


def demonstrate_routing_to_end():
    """Show conditional routing where END is a possible target."""
    print("\n" + "=" * 60)
    print("PART 3: Routing to END (Short-Circuit Pattern)")
    print("=" * 60)
    print()
    print("A routing function can return END to terminate the graph early.")
    print("This is useful for validation pipelines where invalid input")
    print("should skip all remaining processing.")
    print()
    print("Pattern: validate -> (process | END)")
    print()

    builder = StateGraph(ValidationState)

    builder.add_node("validate", validate_input)
    builder.add_node("process", process_data)

    builder.add_edge(START, "validate")

    # Conditional edge that can route to END
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "process": "process",
            "end": END,  # Short-circuit to END
        }
    )

    builder.add_edge("process", END)

    graph = builder.compile()

    # Test with valid and invalid inputs
    test_cases = [
        {"data": "Hello World", "is_valid": False, "error_message": "", "processed_data": "", "validation_steps": []},
        {"data": "Hi", "is_valid": False, "error_message": "", "processed_data": "", "validation_steps": []},
        {"data": "test@data", "is_valid": False, "error_message": "", "processed_data": "", "validation_steps": []},
        {"data": "", "is_valid": False, "error_message": "", "processed_data": "", "validation_steps": []},
    ]

    for case in test_cases:
        print(f"\n  Input: '{case['data']}'")
        result = graph.invoke(case)
        print(f"  -> Valid: {result['is_valid']}")
        if result['is_valid']:
            print(f"  -> Processed: '{result['processed_data']}'")
        else:
            print(f"  -> Error: '{result['error_message']}'")
            print(f"  -> (Processing was SKIPPED - graph short-circuited to END)")
        print(f"  -> Steps: {result['validation_steps']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all conditional routing demonstrations."""
    print("=" * 60)
    print("  LangGraph: Conditional Routing")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates conditional routing in LangGraph.")
    print("No LLM calls are needed - routing is based on state inspection.")

    # Part 1: Basic conditional routing with support tickets
    demonstrate_basic_conditional_routing()

    # Part 2: Multi-condition routing with order processing
    demonstrate_multi_condition_routing()

    # Part 3: Routing to END (short-circuit pattern)
    demonstrate_routing_to_end()

    print("\n" + "=" * 60)
    print("  Exercise 17 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. add_conditional_edges() enables branching based on state")
    print("  2. The routing function receives state and returns a node name string")
    print("  3. The mapping dict maps return values to actual node names")
    print("  4. Multiple conditions can be checked in a single routing function")
    print("  5. END can be a routing target to short-circuit the graph")
    print("  6. Conditional routing is deterministic - same state = same route")
    print("  7. This pattern is ideal when routing logic is rule-based")
    print("  8. For LLM-based routing decisions, see Exercise 18 (Dynamic Routing)")
    print()


if __name__ == "__main__":
    main()
