"""
Module 1 - Exercise 04: Chains and Pipelines with LCEL
========================================================
Learn how to compose chains using LangChain Expression Language (LCEL).
The pipe operator (|) allows you to connect components into pipelines
where the output of one step flows into the next.

Concepts covered:
- LCEL (LangChain Expression Language) and the pipe operator (|)
- Basic chain: prompt | llm | output_parser
- Sequential chains: output of one chain feeds into the next
- RunnablePassthrough to pass data through unchanged
- RunnableLambda for custom transformations in a chain
- RunnableParallel to run multiple chains simultaneously and combine results
"""

import sys
sys.path.append('..')
from config import load_config, get_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)


# ---------------------------------------------------------------------------
# Part 1: Basic Chain with prompt | llm | StrOutputParser
# ---------------------------------------------------------------------------

def demonstrate_basic_chain(llm):
    """Show the fundamental LCEL pattern: prompt | llm | parser."""
    print("\n" + "=" * 60)
    print("PART 1: Basic Chain (prompt | llm | StrOutputParser)")
    print("=" * 60)
    print()
    print("LCEL uses the pipe operator (|) to compose components into chains.")
    print("Each component's output becomes the next component's input:")
    print()
    print("  prompt template  -->  LLM  -->  output parser")
    print("  (formats input)     (generates)  (extracts string)")
    print()
    print("This is the most common pattern in LangChain applications.")
    print()

    # Define a prompt template
    prompt = ChatPromptTemplate.from_template(
        "Explain the concept of '{topic}' in exactly 2 sentences."
    )

    # Create the chain using the pipe operator
    chain = prompt | llm | StrOutputParser()

    print("Chain created: prompt | llm | StrOutputParser()")
    print("-" * 40)
    print()

    # Invoke the chain with input
    topic = "recursion in programming"
    print(f"Input: topic = '{topic}'")
    print()

    result = chain.invoke({"topic": topic})

    print(f"Output: {result}")
    print()
    print("How it works step by step:")
    print("  1. prompt.invoke({'topic': 'recursion...'}) -> ChatPromptValue")
    print("  2. llm.invoke(ChatPromptValue) -> AIMessage")
    print("  3. StrOutputParser().invoke(AIMessage) -> plain string")
    print()

    # Show that the chain can be reused with different inputs
    print("Reusing the same chain with a different input:")
    print("-" * 40)
    topic2 = "API rate limiting"
    result2 = chain.invoke({"topic": topic2})
    print(f"Input: topic = '{topic2}'")
    print(f"Output: {result2}")


# ---------------------------------------------------------------------------
# Part 2: Sequential Chains (output of chain 1 feeds into chain 2)
# ---------------------------------------------------------------------------

def demonstrate_sequential_chains(llm):
    """Show how to chain multiple steps where output feeds into the next."""
    print("\n" + "=" * 60)
    print("PART 2: Sequential Chains (Multi-Step Pipelines)")
    print("=" * 60)
    print()
    print("Sequential chains pass the output of one chain as input to the next.")
    print("Example: Generate a topic -> Write a short poem about that topic.")
    print()
    print("This is useful when you need multi-step reasoning or when one LLM")
    print("call needs to build on the result of a previous one.")
    print()

    # Chain 1: Generate a creative topic
    prompt_topic = ChatPromptTemplate.from_template(
        "Generate a single creative topic for a haiku poem about nature. "
        "Reply with ONLY the topic (2-4 words), nothing else."
    )
    chain_topic = prompt_topic | llm | StrOutputParser()

    # Chain 2: Write a haiku about the topic
    prompt_poem = ChatPromptTemplate.from_template(
        "Write a haiku (3 lines: 5-7-5 syllables) about: {topic}\n"
        "Reply with ONLY the haiku, no title or explanation."
    )
    chain_poem = prompt_poem | llm | StrOutputParser()

    # Execute sequentially: chain 1 output -> chain 2 input
    print("Step 1: Generating a creative topic...")
    print("-" * 40)
    topic = chain_topic.invoke({})
    print(f"Generated topic: '{topic.strip()}'")
    print()

    print("Step 2: Writing a haiku about that topic...")
    print("-" * 40)
    poem = chain_poem.invoke({"topic": topic.strip()})
    print(f"Haiku:\n{poem}")
    print()

    # Now combine them into a single composed chain using RunnableLambda
    print("Combining into a single composed chain:")
    print("-" * 40)
    print()

    # Use a lambda to bridge the two chains
    composed_chain = (
        prompt_topic
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda topic_text: {"topic": topic_text.strip()})
        | prompt_poem
        | llm
        | StrOutputParser()
    )

    result = composed_chain.invoke({})
    print(f"Single-chain result (topic + haiku in one call):\n{result}")
    print()
    print("The composed chain runs both steps automatically in sequence.")
    print("RunnableLambda bridges the gap by transforming the string output")
    print("into the dict format expected by the next prompt template.")


# ---------------------------------------------------------------------------
# Part 3: RunnablePassthrough (pass data through unchanged)
# ---------------------------------------------------------------------------

def demonstrate_passthrough(llm):
    """Show how RunnablePassthrough passes data alongside transformations."""
    print("\n" + "=" * 60)
    print("PART 3: RunnablePassthrough (Pass Data Through Unchanged)")
    print("=" * 60)
    print()
    print("RunnablePassthrough passes its input through without modification.")
    print("This is useful when you need the original input alongside computed")
    print("values (e.g., keeping the question while adding retrieved context).")
    print()
    print("Common pattern in RAG:")
    print("  {'context': retriever, 'question': RunnablePassthrough()}")
    print()

    # Simulate a context retrieval function
    def fake_retriever(query: str) -> str:
        """Simulate document retrieval (in real apps, this queries a vector store)."""
        knowledge_base = {
            "python": "Python is a high-level programming language created by Guido van Rossum in 1991. "
                      "It emphasizes code readability and supports multiple paradigms.",
            "langchain": "LangChain is a framework for building applications with LLMs. "
                         "It provides tools for prompts, chains, agents, and memory.",
        }
        # Simple keyword matching for demonstration
        for key, value in knowledge_base.items():
            if key in query.lower():
                return value
        return "No relevant context found."

    # Create a chain that uses both the original question AND retrieved context
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context below.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # RunnablePassthrough passes the input dict through, while we add context
    chain = (
        RunnableParallel(
            context=RunnableLambda(lambda x: fake_retriever(x["question"])),
            question=RunnableLambda(lambda x: x["question"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What is Python and who created it?"
    print(f"Question: '{question}'")
    print("-" * 40)
    print()

    # Show what RunnablePassthrough does
    print("What happens internally:")
    context = fake_retriever(question)
    print(f"  context (retrieved): '{context[:60]}...'")
    print(f"  question (passed through): '{question}'")
    print()

    result = chain.invoke({"question": question})
    print(f"Answer: {result}")
    print()

    # Demonstrate with RunnablePassthrough.assign() to add fields
    print("Using RunnablePassthrough.assign() to ADD fields to existing data:")
    print("-" * 40)
    print()

    chain_with_assign = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: fake_retriever(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    question2 = "What is LangChain used for?"
    print(f"Input: {{'question': '{question2}'}}")
    print()
    print("RunnablePassthrough.assign(context=...) keeps 'question' and ADDS 'context'")
    print()

    result2 = chain_with_assign.invoke({"question": question2})
    print(f"Answer: {result2}")


# ---------------------------------------------------------------------------
# Part 4: RunnableParallel (run multiple chains simultaneously)
# ---------------------------------------------------------------------------

def demonstrate_parallel(llm):
    """Show how RunnableParallel runs multiple chains and combines results."""
    print("\n" + "=" * 60)
    print("PART 4: RunnableParallel (Run Multiple Chains Simultaneously)")
    print("=" * 60)
    print()
    print("RunnableParallel executes multiple runnables at the same time")
    print("and returns a dict with all results. This is useful when you need")
    print("multiple independent analyses of the same input.")
    print()
    print("Example: Analyze a topic from multiple perspectives in parallel.")
    print()

    # Define three different analysis prompts
    prompt_pros = ChatPromptTemplate.from_template(
        "List 2 advantages of {topic}. Be concise (one line each)."
    )
    prompt_cons = ChatPromptTemplate.from_template(
        "List 2 disadvantages of {topic}. Be concise (one line each)."
    )
    prompt_summary = ChatPromptTemplate.from_template(
        "Write a one-sentence neutral summary of {topic}."
    )

    # Create individual chains
    chain_pros = prompt_pros | llm | StrOutputParser()
    chain_cons = prompt_cons | llm | StrOutputParser()
    chain_summary = prompt_summary | llm | StrOutputParser()

    # Combine them with RunnableParallel
    parallel_chain = RunnableParallel(
        pros=chain_pros,
        cons=chain_cons,
        summary=chain_summary,
    )

    topic = "remote work"
    print(f"Topic: '{topic}'")
    print(f"Running 3 chains in parallel...")
    print("-" * 40)
    print()

    # All three chains run simultaneously
    results = parallel_chain.invoke({"topic": topic})

    print("Results (all computed in parallel):")
    print()
    print(f"  SUMMARY:")
    print(f"    {results['summary']}")
    print()
    print(f"  PROS:")
    for line in results['pros'].strip().split('\n'):
        print(f"    {line}")
    print()
    print(f"  CONS:")
    for line in results['cons'].strip().split('\n'):
        print(f"    {line}")
    print()

    # Show that RunnableParallel can also be used for input preparation
    print("RunnableParallel for input preparation:")
    print("-" * 40)
    print()
    print("RunnableParallel is also commonly used to prepare multiple inputs")
    print("for a downstream prompt (as seen in Part 3 with context + question).")
    print()

    # Another example: generate a comparison
    prompt_compare = ChatPromptTemplate.from_template(
        "Given these perspectives on {topic}:\n"
        "Pros: {pros}\n"
        "Cons: {cons}\n\n"
        "Write a balanced 2-sentence conclusion."
    )

    # Full pipeline: parallel analysis -> comparison
    full_chain = (
        RunnableParallel(
            topic=RunnableLambda(lambda x: x["topic"]),
            pros=chain_pros,
            cons=chain_cons,
        )
        | prompt_compare
        | llm
        | StrOutputParser()
    )

    print("Full pipeline: parallel(pros, cons) -> comparison prompt -> LLM")
    print()
    conclusion = full_chain.invoke({"topic": topic})
    print(f"Balanced conclusion: {conclusion}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all LCEL chain demonstrations."""
    print("=" * 60)
    print("  LangChain Fundamentals: Chains and Pipelines with LCEL")
    print("=" * 60)

    # Load configuration and initialize LLM
    config = load_config()
    llm = get_llm()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Temperature: {config['default_temperature']}")

    # Part 1: Basic chain with pipe operator
    demonstrate_basic_chain(llm)

    # Part 2: Sequential chains (output feeds into next)
    demonstrate_sequential_chains(llm)

    # Part 3: RunnablePassthrough for passing data alongside transformations
    demonstrate_passthrough(llm)

    # Part 4: RunnableParallel for running multiple chains simultaneously
    demonstrate_parallel(llm)

    print("\n" + "=" * 60)
    print("  Exercise 04 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. LCEL pipe operator (|) composes components into chains")
    print("  2. Basic pattern: prompt | llm | StrOutputParser()")
    print("  3. Sequential chains: output of one step feeds into the next")
    print("  4. RunnableLambda: custom transformations between chain steps")
    print("  5. RunnablePassthrough: pass input through unchanged (useful in RAG)")
    print("  6. RunnablePassthrough.assign(): add computed fields to existing data")
    print("  7. RunnableParallel: run multiple chains simultaneously, combine results")
    print("  8. These primitives compose together for complex pipelines")
    print()


if __name__ == "__main__":
    main()
