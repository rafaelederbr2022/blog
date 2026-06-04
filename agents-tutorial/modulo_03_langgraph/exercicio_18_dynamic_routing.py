"""
Module 3 - Exercise 18: Dynamic Routing in LangGraph
=====================================================
Learn how to use an LLM to make routing decisions in a LangGraph workflow.
Unlike conditional routing (Exercise 17) where rules are hardcoded,
dynamic routing lets the LLM analyze input and decide which path to take.

Concepts covered:
- Using an LLM to classify input and determine the next node
- Combining LLM-based routing with add_conditional_edges()
- Structured output from LLM for reliable routing decisions
- Fallback handling when LLM returns unexpected values
- Real-world pattern: query classifier that routes to specialist nodes

Example: A query classifier that uses the LLM to determine whether a user
question is about coding, math, or creative writing, then routes to the
appropriate specialist node for handling.
"""

import sys
sys.path.append('..')
from config import load_config, get_llm

from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Part 1: LLM-Based Query Classification and Routing
# ---------------------------------------------------------------------------

class QueryState(TypedDict):
    """State for the query routing system.

    The LLM classifies the query and the graph routes accordingly.
    """
    query: str
    classification: str
    specialist_response: str
    confidence: str
    routing_log: Annotated[list[str], operator.add]


# Pydantic model for structured LLM output
class QueryClassification(BaseModel):
    """Structured output for query classification."""
    category: Literal["coding", "math", "creative", "general"] = Field(
        description="The category of the user's query"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the classification"
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )


def classify_with_llm(state: QueryState) -> dict:
    """Use the LLM to classify the user's query into a category.

    This is the key difference from conditional routing:
    the routing decision is made by the MODEL, not by hardcoded rules.
    """
    print(f"  [classify_with_llm] Query: '{state['query']}'")
    print(f"  [classify_with_llm] Asking LLM to classify...")

    llm = get_llm(temperature=0)

    # Use structured output for reliable classification
    structured_llm = llm.with_structured_output(QueryClassification)

    messages = [
        SystemMessage(content=(
            "You are a query classifier. Classify the user's query into exactly one category:\n"
            "- 'coding': programming, software development, debugging, algorithms\n"
            "- 'math': mathematics, calculations, statistics, equations\n"
            "- 'creative': creative writing, storytelling, poetry, brainstorming\n"
            "- 'general': anything that doesn't fit the above categories\n\n"
            "Provide your classification with confidence level and brief reasoning."
        )),
        HumanMessage(content=state["query"]),
    ]

    result = structured_llm.invoke(messages)

    print(f"  [classify_with_llm] Classification: {result.category} (confidence: {result.confidence})")
    print(f"  [classify_with_llm] Reasoning: {result.reasoning}")

    return {
        "classification": result.category,
        "confidence": result.confidence,
        "routing_log": [
            f"LLM classified as '{result.category}' ({result.confidence} confidence)",
            f"Reasoning: {result.reasoning}",
        ],
    }


def route_by_classification(state: QueryState) -> str:
    """Route to the appropriate specialist based on LLM classification.

    This routing function simply reads the classification that the LLM
    already determined. The intelligence is in the classify node.
    """
    classification = state["classification"]
    print(f"  [router] Routing to '{classification}' specialist...")
    return classification


def coding_specialist(state: QueryState) -> dict:
    """Handle coding-related queries with a specialized prompt."""
    print(f"  [coding_specialist] Handling coding query...")

    llm = get_llm(temperature=0.3)

    messages = [
        SystemMessage(content=(
            "You are an expert software engineer. Provide clear, concise answers "
            "to coding questions. Include code examples when helpful. "
            "Keep your response under 3 sentences for this demo."
        )),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    print(f"  [coding_specialist] Response: {response.content[:100]}...")

    return {
        "specialist_response": response.content,
        "routing_log": ["Handled by coding specialist"],
    }


def math_specialist(state: QueryState) -> dict:
    """Handle math-related queries with a specialized prompt."""
    print(f"  [math_specialist] Handling math query...")

    llm = get_llm(temperature=0)

    messages = [
        SystemMessage(content=(
            "You are an expert mathematician. Provide clear, step-by-step "
            "explanations for math problems. Show your work. "
            "Keep your response under 3 sentences for this demo."
        )),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    print(f"  [math_specialist] Response: {response.content[:100]}...")

    return {
        "specialist_response": response.content,
        "routing_log": ["Handled by math specialist"],
    }


def creative_specialist(state: QueryState) -> dict:
    """Handle creative writing queries with a specialized prompt."""
    print(f"  [creative_specialist] Handling creative query...")

    llm = get_llm(temperature=0.9)

    messages = [
        SystemMessage(content=(
            "You are a creative writing expert. Be imaginative, expressive, "
            "and inspiring in your responses. Use vivid language. "
            "Keep your response under 3 sentences for this demo."
        )),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    print(f"  [creative_specialist] Response: {response.content[:100]}...")

    return {
        "specialist_response": response.content,
        "routing_log": ["Handled by creative specialist"],
    }


def general_specialist(state: QueryState) -> dict:
    """Handle general queries that don't fit other categories."""
    print(f"  [general_specialist] Handling general query...")

    llm = get_llm(temperature=0.5)

    messages = [
        SystemMessage(content=(
            "You are a helpful general assistant. Provide clear and informative "
            "answers. Keep your response under 3 sentences for this demo."
        )),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    print(f"  [general_specialist] Response: {response.content[:100]}...")

    return {
        "specialist_response": response.content,
        "routing_log": ["Handled by general specialist"],
    }


def demonstrate_llm_routing():
    """Show LLM-based dynamic routing with query classification."""
    print("\n" + "=" * 60)
    print("PART 1: LLM-Based Dynamic Routing - Query Classifier")
    print("=" * 60)
    print()
    print("In dynamic routing, the LLM decides which path to take.")
    print("The graph structure is:")
    print()
    print("  START -> classify_with_llm -> [router] -> specialist -> END")
    print("                                    |")
    print("                          coding / math / creative / general")
    print()
    print("The LLM analyzes the query and returns a structured classification.")
    print("The routing function then uses that classification to pick the next node.")
    print()

    # Build the graph
    builder = StateGraph(QueryState)

    # Add nodes
    builder.add_node("classify", classify_with_llm)
    builder.add_node("coding", coding_specialist)
    builder.add_node("math", math_specialist)
    builder.add_node("creative", creative_specialist)
    builder.add_node("general", general_specialist)

    # Fixed edge: START -> classify
    builder.add_edge(START, "classify")

    # Dynamic routing: classify -> (coding | math | creative | general)
    # The LLM's classification determines which specialist handles the query
    builder.add_conditional_edges(
        "classify",
        route_by_classification,
        {
            "coding": "coding",
            "math": "math",
            "creative": "creative",
            "general": "general",
        }
    )

    # All specialists lead to END
    builder.add_edge("coding", END)
    builder.add_edge("math", END)
    builder.add_edge("creative", END)
    builder.add_edge("general", END)

    graph = builder.compile()

    # Test with different query types
    queries = [
        "How do I reverse a linked list in Python?",
        "What is the derivative of x^3 + 2x?",
        "Write me a haiku about the ocean at sunset",
    ]

    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 50}")
        print()

        result = graph.invoke({
            "query": query,
            "classification": "",
            "specialist_response": "",
            "confidence": "",
            "routing_log": [],
        })

        print()
        print(f"  Classification: {result['classification']} (confidence: {result['confidence']})")
        print(f"  Specialist Response: {result['specialist_response'][:150]}...")
        print(f"  Routing Log: {result['routing_log']}")


# ---------------------------------------------------------------------------
# Part 2: Dynamic Routing with Confidence-Based Fallback
# ---------------------------------------------------------------------------

class AdvancedQueryState(TypedDict):
    """State with confidence-based routing logic."""
    query: str
    classification: str
    confidence: str
    needs_human_review: bool
    response: str
    routing_log: Annotated[list[str], operator.add]


def classify_with_confidence(state: AdvancedQueryState) -> dict:
    """Classify query and track confidence for routing decisions."""
    print(f"  [classify_with_confidence] Query: '{state['query']}'")

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(QueryClassification)

    messages = [
        SystemMessage(content=(
            "You are a query classifier. Classify the user's query into exactly one category:\n"
            "- 'coding': programming, software development, debugging\n"
            "- 'math': mathematics, calculations, statistics\n"
            "- 'creative': creative writing, storytelling, poetry\n"
            "- 'general': anything else\n\n"
            "Be honest about your confidence. Use 'low' if the query is ambiguous "
            "or could fit multiple categories."
        )),
        HumanMessage(content=state["query"]),
    ]

    result = structured_llm.invoke(messages)

    print(f"  [classify_with_confidence] Result: {result.category} ({result.confidence})")

    return {
        "classification": result.category,
        "confidence": result.confidence,
        "needs_human_review": result.confidence == "low",
        "routing_log": [f"Classified: {result.category} (confidence: {result.confidence})"],
    }


def route_with_confidence_check(state: AdvancedQueryState) -> str:
    """Route based on classification, but flag low-confidence for human review.

    This demonstrates combining LLM routing with confidence thresholds.
    Low-confidence classifications get routed to a human review node.
    """
    if state["confidence"] == "low":
        print(f"  [router] Low confidence -> routing to human review")
        return "human_review"
    else:
        print(f"  [router] Sufficient confidence -> routing to auto_respond")
        return "auto_respond"


def auto_respond(state: AdvancedQueryState) -> dict:
    """Automatically respond when confidence is sufficient."""
    print(f"  [auto_respond] Generating response for '{state['classification']}' query...")

    llm = get_llm(temperature=0.5)

    messages = [
        SystemMessage(content=f"You are a {state['classification']} expert. Answer concisely in 1-2 sentences."),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    print(f"  [auto_respond] Response generated.")

    return {
        "response": response.content,
        "routing_log": [f"Auto-responded as {state['classification']} expert"],
    }


def human_review(state: AdvancedQueryState) -> dict:
    """Flag for human review when confidence is low.

    In a real system, this would pause execution and wait for human input.
    Here we simulate the human review process.
    """
    print(f"  [human_review] LOW CONFIDENCE - Flagging for human review")
    print(f"  [human_review] Query: '{state['query']}'")
    print(f"  [human_review] LLM suggested: '{state['classification']}' but was unsure")

    # Simulate human review (in production, this would pause for real input)
    response = (
        f"[HUMAN REVIEW REQUIRED] Query '{state['query'][:40]}...' was classified as "
        f"'{state['classification']}' with low confidence. A human agent should verify "
        f"the classification and provide a response."
    )

    return {
        "response": response,
        "routing_log": ["Flagged for human review due to low confidence"],
    }


def demonstrate_confidence_routing():
    """Show dynamic routing with confidence-based fallback."""
    print("\n" + "=" * 60)
    print("PART 2: Confidence-Based Dynamic Routing")
    print("=" * 60)
    print()
    print("This pattern adds a confidence check after LLM classification.")
    print("High/medium confidence queries are auto-handled.")
    print("Low confidence queries are flagged for human review.")
    print()
    print("  START -> classify -> [confidence check] -> auto_respond -> END")
    print("                              |")
    print("                        (low confidence)")
    print("                              |")
    print("                        human_review -> END")
    print()

    # Build the graph
    builder = StateGraph(AdvancedQueryState)

    builder.add_node("classify", classify_with_confidence)
    builder.add_node("auto_respond", auto_respond)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        route_with_confidence_check,
        {
            "auto_respond": "auto_respond",
            "human_review": "human_review",
        }
    )

    builder.add_edge("auto_respond", END)
    builder.add_edge("human_review", END)

    graph = builder.compile()

    # Test with clear and ambiguous queries
    queries = [
        "What is a Python decorator?",  # Clear coding question
        "Tell me something interesting about the number 7 in literature and math",  # Ambiguous
    ]

    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 50}")
        print()

        result = graph.invoke({
            "query": query,
            "classification": "",
            "confidence": "",
            "needs_human_review": False,
            "response": "",
            "routing_log": [],
        })

        print()
        print(f"  Classification: {result['classification']} (confidence: {result['confidence']})")
        print(f"  Needs Human Review: {result['needs_human_review']}")
        print(f"  Response: {result['response'][:120]}...")
        print(f"  Routing Log: {result['routing_log']}")


# ---------------------------------------------------------------------------
# Part 3: Multi-Step Dynamic Routing (Classify -> Refine -> Route)
# ---------------------------------------------------------------------------

class IntentState(TypedDict):
    """State for multi-step intent classification and routing."""
    user_input: str
    intent: str
    sub_intent: str
    final_response: str
    routing_log: Annotated[list[str], operator.add]


# Pydantic models for structured outputs
class IntentClassification(BaseModel):
    """First-level intent classification."""
    intent: Literal["question", "action", "feedback"] = Field(
        description="The primary intent of the user's input"
    )
    reasoning: str = Field(description="Why this intent was chosen")


class ActionSubIntent(BaseModel):
    """Sub-classification for action intents."""
    sub_intent: Literal["create", "update", "delete"] = Field(
        description="The specific action the user wants to perform"
    )
    target: str = Field(description="What the user wants to act on")


def classify_intent(state: IntentState) -> dict:
    """First LLM call: classify the primary intent."""
    print(f"  [classify_intent] Input: '{state['user_input']}'")

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(IntentClassification)

    messages = [
        SystemMessage(content=(
            "Classify the user's input into one of these intents:\n"
            "- 'question': user is asking for information\n"
            "- 'action': user wants to perform an action (create, update, delete something)\n"
            "- 'feedback': user is providing feedback or a comment"
        )),
        HumanMessage(content=state["user_input"]),
    ]

    result = structured_llm.invoke(messages)
    print(f"  [classify_intent] Intent: {result.intent} ({result.reasoning})")

    return {
        "intent": result.intent,
        "routing_log": [f"Primary intent: {result.intent}"],
    }


def route_by_intent(state: IntentState) -> str:
    """Route based on primary intent classification."""
    intent = state["intent"]
    print(f"  [router] Primary intent: '{intent}'")
    return intent


def handle_question(state: IntentState) -> dict:
    """Handle question intents."""
    print(f"  [handle_question] Answering question...")

    llm = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content="Answer the user's question concisely in 1-2 sentences."),
        HumanMessage(content=state["user_input"]),
    ]
    response = llm.invoke(messages)

    return {
        "final_response": f"[ANSWER] {response.content}",
        "routing_log": ["Handled as question"],
    }


def refine_action_intent(state: IntentState) -> dict:
    """Second LLM call: refine the action intent into a sub-category.

    This demonstrates MULTI-STEP dynamic routing where the LLM is called
    multiple times to progressively narrow down the routing decision.
    """
    print(f"  [refine_action_intent] Refining action intent...")

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ActionSubIntent)

    messages = [
        SystemMessage(content=(
            "The user wants to perform an action. Classify the specific action type:\n"
            "- 'create': user wants to create or add something new\n"
            "- 'update': user wants to modify or change something existing\n"
            "- 'delete': user wants to remove or delete something"
        )),
        HumanMessage(content=state["user_input"]),
    ]

    result = structured_llm.invoke(messages)
    print(f"  [refine_action_intent] Sub-intent: {result.sub_intent} (target: {result.target})")

    return {
        "sub_intent": result.sub_intent,
        "routing_log": [f"Action refined: {result.sub_intent} on '{result.target}'"],
    }


def route_by_sub_intent(state: IntentState) -> str:
    """Route based on refined sub-intent."""
    sub_intent = state["sub_intent"]
    print(f"  [router] Sub-intent: '{sub_intent}'")
    return sub_intent


def handle_create(state: IntentState) -> dict:
    """Handle create actions."""
    print(f"  [handle_create] Processing create action...")
    return {
        "final_response": f"[CREATE] Initiating creation process for: '{state['user_input']}'",
        "routing_log": ["Handled as create action"],
    }


def handle_update(state: IntentState) -> dict:
    """Handle update actions."""
    print(f"  [handle_update] Processing update action...")
    return {
        "final_response": f"[UPDATE] Initiating update process for: '{state['user_input']}'",
        "routing_log": ["Handled as update action"],
    }


def handle_delete(state: IntentState) -> dict:
    """Handle delete actions."""
    print(f"  [handle_delete] Processing delete action...")
    return {
        "final_response": f"[DELETE] Initiating delete process for: '{state['user_input']}'",
        "routing_log": ["Handled as delete action"],
    }


def handle_feedback(state: IntentState) -> dict:
    """Handle feedback intents."""
    print(f"  [handle_feedback] Processing feedback...")
    return {
        "final_response": f"[FEEDBACK] Thank you for your feedback: '{state['user_input'][:50]}...'",
        "routing_log": ["Handled as feedback"],
    }


def demonstrate_multi_step_routing():
    """Show multi-step dynamic routing with progressive LLM classification."""
    print("\n" + "=" * 60)
    print("PART 3: Multi-Step Dynamic Routing")
    print("=" * 60)
    print()
    print("This pattern uses MULTIPLE LLM calls to progressively refine")
    print("the routing decision. First classify the intent, then for")
    print("'action' intents, classify the specific action type.")
    print()
    print("  START -> classify_intent -> [intent router]")
    print("                                  |")
    print("                    question / action / feedback")
    print("                        |        |         |")
    print("                   answer   refine_action  handle_feedback -> END")
    print("                               |")
    print("                    [sub-intent router]")
    print("                     /      |       \\")
    print("                 create  update   delete -> END")
    print()

    # Build the graph
    builder = StateGraph(IntentState)

    # Add all nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("handle_question", handle_question)
    builder.add_node("refine_action", refine_action_intent)
    builder.add_node("handle_create", handle_create)
    builder.add_node("handle_update", handle_update)
    builder.add_node("handle_delete", handle_delete)
    builder.add_node("handle_feedback", handle_feedback)

    # Entry
    builder.add_edge(START, "classify_intent")

    # First routing: by primary intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "question": "handle_question",
            "action": "refine_action",
            "feedback": "handle_feedback",
        }
    )

    # Second routing: by sub-intent (only for action intents)
    builder.add_conditional_edges(
        "refine_action",
        route_by_sub_intent,
        {
            "create": "handle_create",
            "update": "handle_update",
            "delete": "handle_delete",
        }
    )

    # All terminal nodes lead to END
    builder.add_edge("handle_question", END)
    builder.add_edge("handle_create", END)
    builder.add_edge("handle_update", END)
    builder.add_edge("handle_delete", END)
    builder.add_edge("handle_feedback", END)

    graph = builder.compile()

    # Test with different input types
    inputs = [
        "What is the capital of France?",
        "Please create a new project called 'AI Research'",
        "Delete my old backup files from last month",
        "The new dashboard design looks great, nice work!",
    ]

    for user_input in inputs:
        print(f"\n{'─' * 50}")
        print(f"  Input: \"{user_input}\"")
        print(f"{'─' * 50}")
        print()

        result = graph.invoke({
            "user_input": user_input,
            "intent": "",
            "sub_intent": "",
            "final_response": "",
            "routing_log": [],
        })

        print()
        print(f"  Intent: {result['intent']}")
        if result['sub_intent']:
            print(f"  Sub-Intent: {result['sub_intent']}")
        print(f"  Response: {result['final_response'][:100]}...")
        print(f"  Routing Log: {result['routing_log']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all dynamic routing demonstrations."""
    print("=" * 60)
    print("  LangGraph: Dynamic Routing (LLM-Based)")
    print("=" * 60)

    # Load configuration (validates environment)
    load_config()

    print("\nThis exercise demonstrates dynamic routing where the LLM")
    print("decides which path the graph should take.")
    print()
    print("Key difference from Exercise 17 (Conditional Routing):")
    print("  - Conditional: routing logic is hardcoded in Python functions")
    print("  - Dynamic: routing logic is determined by the LLM at runtime")
    print()
    print("NOTE: This exercise makes LLM API calls.")

    # Part 1: Basic LLM-based query classification and routing
    demonstrate_llm_routing()

    # Part 2: Confidence-based routing with human review fallback
    demonstrate_confidence_routing()

    # Part 3: Multi-step dynamic routing with progressive classification
    demonstrate_multi_step_routing()

    print("\n" + "=" * 60)
    print("  Exercise 18 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Dynamic routing uses the LLM to decide the graph's path")
    print("  2. Structured output (Pydantic) ensures reliable routing values")
    print("  3. The classify node sets state, the router reads it")
    print("  4. Confidence levels enable fallback to human review")
    print("  5. Multi-step routing progressively narrows the decision")
    print("  6. Each specialist node can use different LLM settings (temperature)")
    print("  7. Combine with conditional edges for hybrid routing strategies")
    print("  8. Always handle unexpected classifications with a fallback path")
    print()


if __name__ == "__main__":
    main()
