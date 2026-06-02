"""
Module 1 - Exercise 01: Prompt Templates & Structured Output
=============================================================
Learn how to create reusable prompt templates with dynamic variables
and get structured, validated output from LLMs using Pydantic schemas.

Concepts covered:
- PromptTemplate with variables (basic string formatting)
- ChatPromptTemplate with message roles (system/human)
- Structured Output with Pydantic validation (.with_structured_output())
"""

import sys
sys.path.append('..')
from config import load_config, get_llm

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic model for Structured Output
# ---------------------------------------------------------------------------

class MovieReview(BaseModel):
    """Schema for a structured movie review response."""
    title: str = Field(description="The title of the movie")
    rating: float = Field(description="Rating from 0 to 10")
    summary: str = Field(description="A brief summary of the movie plot")
    strengths: list[str] = Field(description="List of movie strengths")
    weaknesses: list[str] = Field(description="List of movie weaknesses")


# ---------------------------------------------------------------------------
# Part 1: PromptTemplate with dynamic variables
# ---------------------------------------------------------------------------

def demonstrate_prompt_template(llm):
    """Demonstrate basic PromptTemplate with variable substitution."""
    print("\n" + "=" * 60)
    print("PART 1: PromptTemplate with Dynamic Variables")
    print("=" * 60)
    print()
    print("PromptTemplate allows you to create reusable templates with")
    print("placeholders that get filled in at runtime.")
    print()

    # Create a simple PromptTemplate with one variable
    template = PromptTemplate(
        input_variables=["topic"],
        template="Provide a concise summary (3-4 sentences) about: {topic}"
    )

    # Show the template structure
    print(f"Template: '{template.template}'")
    print(f"Variables: {template.input_variables}")
    print()

    # Format the template with a value
    topic = "the history of artificial intelligence"
    formatted_prompt = template.format(topic=topic)
    print(f"Formatted prompt: '{formatted_prompt}'")
    print()

    # Invoke the LLM with the formatted prompt
    print("Sending to LLM...")
    print("-" * 40)
    response = llm.invoke(formatted_prompt)
    print(f"Response:\n{response.content}")
    print()

    # Demonstrate with a different variable value
    topic2 = "quantum computing applications"
    formatted_prompt2 = template.format(topic=topic2)
    print(f"Same template, different topic: '{topic2}'")
    print("-" * 40)
    response2 = llm.invoke(formatted_prompt2)
    print(f"Response:\n{response2.content}")


# ---------------------------------------------------------------------------
# Part 2: ChatPromptTemplate with system/human messages
# ---------------------------------------------------------------------------

def demonstrate_chat_prompt_template(llm):
    """Demonstrate ChatPromptTemplate with system and human message roles."""
    print("\n" + "=" * 60)
    print("PART 2: ChatPromptTemplate with Message Roles")
    print("=" * 60)
    print()
    print("ChatPromptTemplate lets you define multi-message prompts with")
    print("different roles (system, human, ai). The system message sets")
    print("the behavior, and the human message provides the query.")
    print()

    # Create a ChatPromptTemplate with system and human messages
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} who explains concepts in a {style} way. "
                   "Keep responses under 100 words."),
        ("human", "{question}")
    ])

    # Show the template structure
    print("Message roles defined:")
    print("  - system: Sets the AI's role and communication style")
    print("  - human: Contains the user's question")
    print(f"Variables: role, style, question")
    print()

    # Format and invoke with specific values
    role = "science teacher"
    style = "simple and engaging"
    question = "What causes rainbows to appear?"

    messages = chat_template.format_messages(
        role=role,
        style=style,
        question=question
    )

    print(f"Role: '{role}'")
    print(f"Style: '{style}'")
    print(f"Question: '{question}'")
    print("-" * 40)

    response = llm.invoke(messages)
    print(f"Response:\n{response.content}")
    print()

    # Same template with different role and style
    role2 = "stand-up comedian"
    style2 = "humorous and witty"
    question2 = "Why do we need to sleep?"

    messages2 = chat_template.format_messages(
        role=role2,
        style=style2,
        question=question2
    )

    print(f"Same template, different persona:")
    print(f"Role: '{role2}'")
    print(f"Style: '{style2}'")
    print(f"Question: '{question2}'")
    print("-" * 40)

    response2 = llm.invoke(messages2)
    print(f"Response:\n{response2.content}")


# ---------------------------------------------------------------------------
# Part 3: Structured Output with Pydantic validation
# ---------------------------------------------------------------------------

def demonstrate_structured_output(llm):
    """Demonstrate structured output using Pydantic schema validation."""
    print("\n" + "=" * 60)
    print("PART 3: Structured Output with Pydantic Validation")
    print("=" * 60)
    print()
    print("Structured Output forces the LLM to return data matching a")
    print("Pydantic schema. This guarantees type-safe, validated responses")
    print("that you can use directly in your application logic.")
    print()
    print("Schema: MovieReview")
    print("  - title: str")
    print("  - rating: float (0-10)")
    print("  - summary: str")
    print("  - strengths: list[str]")
    print("  - weaknesses: list[str]")
    print()

    # Create a structured LLM using .with_structured_output()
    structured_llm = llm.with_structured_output(MovieReview)

    # Invoke with a prompt asking for a movie review
    prompt = "Give me a detailed review of the movie 'Inception' (2010) by Christopher Nolan."

    print(f"Prompt: '{prompt}'")
    print("-" * 40)
    print("Invoking LLM with structured output...")
    print()

    review = structured_llm.invoke(prompt)

    # Display the structured result
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}/10")
    print(f"Summary: {review.summary}")
    print(f"Strengths:")
    for s in review.strengths:
        print(f"  + {s}")
    print(f"Weaknesses:")
    for w in review.weaknesses:
        print(f"  - {w}")
    print()

    # Show that the result is a proper Pydantic model
    print("Validation proof:")
    print(f"  Type: {type(review).__name__}")
    print(f"  Is MovieReview instance: {isinstance(review, MovieReview)}")
    print(f"  Rating is float: {isinstance(review.rating, float)}")
    print(f"  Strengths is list: {isinstance(review.strengths, list)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all three demonstrations."""
    print("=" * 60)
    print("  LangChain Fundamentals: Prompt Templates & Structured Output")
    print("=" * 60)

    # Load configuration and initialize LLM
    config = load_config()
    llm = get_llm()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Temperature: {config['default_temperature']}")

    # Run demonstrations
    demonstrate_prompt_template(llm)
    demonstrate_chat_prompt_template(llm)
    demonstrate_structured_output(llm)

    print("\n" + "=" * 60)
    print("  Exercise 01 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. PromptTemplate: Reusable templates with {variable} placeholders")
    print("  2. ChatPromptTemplate: Multi-role messages (system/human/ai)")
    print("  3. Structured Output: Type-safe responses via Pydantic schemas")
    print()


if __name__ == "__main__":
    main()
