"""
Module 1 - Exercise 05: Streaming Tokens from LLMs
=====================================================
Learn how to receive LLM responses incrementally using streaming.
Instead of waiting for the entire response, streaming delivers tokens
as they are generated, improving perceived latency and user experience.

Concepts covered:
- Basic streaming with llm.stream() (synchronous)
- Streaming through a chain (prompt | llm | StrOutputParser) with .stream()
- Async streaming with asyncio and .astream()
- Measuring time-to-first-token vs total generation time
"""

import sys
import asyncio
import time

sys.path.append('..')
from config import load_config, get_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Part 1: Basic Streaming with llm.stream()
# ---------------------------------------------------------------------------

def demonstrate_basic_streaming(llm):
    """Show how to stream tokens from an LLM using .stream()."""
    print("\n" + "=" * 60)
    print("PART 1: Basic Streaming with llm.stream()")
    print("=" * 60)
    print()
    print("By default, .invoke() waits for the ENTIRE response before returning.")
    print("With .stream(), you receive tokens (chunks) as they are generated.")
    print("This is useful for chat interfaces where you want to show progress.")
    print()
    print("Each chunk is an AIMessageChunk containing a piece of the response.")
    print()

    prompt_text = "Write a short paragraph (3-4 sentences) about the history of the internet."

    print(f"Prompt: '{prompt_text}'")
    print("-" * 40)
    print()
    print("Streaming response (tokens arrive one by one):")
    print()

    # Stream tokens from the LLM
    full_response = ""
    chunk_count = 0

    for chunk in llm.stream(prompt_text):
        # Each chunk has a .content attribute with the token text
        token = chunk.content
        print(token, end="", flush=True)
        full_response += token
        chunk_count += 1

    print()
    print()
    print(f"Total chunks received: {chunk_count}")
    print(f"Full response length: {len(full_response)} characters")
    print()
    print("Key insight: Each chunk is an AIMessageChunk object.")
    print("The .content attribute contains the actual token text.")
    print("Using flush=True ensures tokens appear immediately in the console.")


# ---------------------------------------------------------------------------
# Part 2: Streaming Through a Chain (prompt | llm | StrOutputParser)
# ---------------------------------------------------------------------------

def demonstrate_chain_streaming(llm):
    """Show how to stream through a complete LCEL chain."""
    print("\n" + "=" * 60)
    print("PART 2: Streaming Through a Chain")
    print("=" * 60)
    print()
    print("You can also stream through a full chain (prompt | llm | parser).")
    print("The StrOutputParser extracts the text content from each chunk,")
    print("so you receive plain strings instead of AIMessageChunk objects.")
    print()
    print("This is the most common pattern for streaming in applications.")
    print()

    # Create a chain with prompt, llm, and output parser
    prompt = ChatPromptTemplate.from_template(
        "Explain {concept} in simple terms, using an analogy. "
        "Keep it to 3-4 sentences."
    )

    chain = prompt | llm | StrOutputParser()

    concept = "how neural networks learn"
    print(f"Chain: prompt | llm | StrOutputParser()")
    print(f"Input: concept = '{concept}'")
    print("-" * 40)
    print()
    print("Streaming through the chain:")
    print()

    # Stream through the chain - StrOutputParser yields plain strings
    full_response = ""
    chunk_count = 0

    for chunk in chain.stream({"concept": concept}):
        # With StrOutputParser, each chunk is already a plain string
        print(chunk, end="", flush=True)
        full_response += chunk
        chunk_count += 1

    print()
    print()
    print(f"Total chunks: {chunk_count}")
    print()
    print("Notice: With StrOutputParser in the chain, each chunk is a plain")
    print("string (not an AIMessageChunk). The parser handles the extraction.")
    print()
    print("Without StrOutputParser, you would need to access .content on each chunk.")
    print("With it, the chain yields ready-to-use text fragments.")


# ---------------------------------------------------------------------------
# Part 3: Async Streaming with .astream()
# ---------------------------------------------------------------------------

async def demonstrate_async_streaming(llm):
    """Show how to use async streaming with .astream()."""
    print("\n" + "=" * 60)
    print("PART 3: Async Streaming with .astream()")
    print("=" * 60)
    print()
    print("For async applications (web servers, APIs), use .astream().")
    print("It returns an async iterator that you consume with 'async for'.")
    print()
    print("Benefits of async streaming:")
    print("  - Non-blocking: other tasks can run while waiting for tokens")
    print("  - Better for web frameworks (FastAPI, aiohttp)")
    print("  - Enables concurrent streaming from multiple LLMs")
    print()

    prompt = ChatPromptTemplate.from_template(
        "List 3 interesting facts about {topic}. One sentence each."
    )

    chain = prompt | llm | StrOutputParser()

    topic = "the ocean"
    print(f"Async streaming about: '{topic}'")
    print("-" * 40)
    print()

    # Use async for with .astream()
    full_response = ""
    chunk_count = 0

    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="", flush=True)
        full_response += chunk
        chunk_count += 1

    print()
    print()
    print(f"Total async chunks: {chunk_count}")
    print()
    print("The async version works identically to the sync version,")
    print("but it allows the event loop to handle other tasks between chunks.")
    print("Use .astream() in async contexts (FastAPI routes, async handlers).")


# ---------------------------------------------------------------------------
# Part 4: Measuring Time-to-First-Token vs Total Generation Time
# ---------------------------------------------------------------------------

def demonstrate_timing_metrics(llm):
    """Measure and compare time-to-first-token (TTFT) vs total time."""
    print("\n" + "=" * 60)
    print("PART 4: Time-to-First-Token vs Total Generation Time")
    print("=" * 60)
    print()
    print("Two important latency metrics for LLM applications:")
    print()
    print("  - Time-to-First-Token (TTFT): How long until the first token arrives.")
    print("    This determines perceived responsiveness.")
    print()
    print("  - Total Generation Time: How long until the full response is complete.")
    print("    This determines overall throughput.")
    print()
    print("Streaming improves TTFT dramatically while total time stays similar.")
    print()

    prompt = ChatPromptTemplate.from_template(
        "Write a brief explanation of {topic} in 4-5 sentences."
    )
    chain = prompt | llm | StrOutputParser()

    topic = "quantum computing"

    # --- Measure streaming timing ---
    print(f"Topic: '{topic}'")
    print()
    print("Measuring with streaming (.stream()):")
    print("-" * 40)

    start_time = time.time()
    first_token_time = None
    full_response = ""
    chunk_count = 0

    for chunk in chain.stream({"topic": topic}):
        if first_token_time is None:
            first_token_time = time.time()
        print(chunk, end="", flush=True)
        full_response += chunk
        chunk_count += 1

    end_time = time.time()

    ttft_streaming = first_token_time - start_time if first_token_time else 0
    total_streaming = end_time - start_time

    print()
    print()
    print(f"  Time-to-First-Token (TTFT): {ttft_streaming:.3f}s")
    print(f"  Total Generation Time:      {total_streaming:.3f}s")
    print(f"  Chunks received:            {chunk_count}")
    print()

    # --- Measure non-streaming timing ---
    print("Measuring without streaming (.invoke()):")
    print("-" * 40)

    start_time = time.time()
    result = chain.invoke({"topic": topic})
    end_time = time.time()

    total_invoke = end_time - start_time

    print(f"  {result[:80]}...")
    print()
    print(f"  Total Time (invoke):        {total_invoke:.3f}s")
    print(f"  Time-to-First-Token:        {total_invoke:.3f}s (same as total - no streaming!)")
    print()

    # --- Comparison ---
    print("Comparison:")
    print("-" * 40)
    print(f"  Streaming TTFT:    {ttft_streaming:.3f}s  (user sees response quickly)")
    print(f"  Invoke TTFT:       {total_invoke:.3f}s  (user waits for everything)")
    print(f"  Streaming Total:   {total_streaming:.3f}s")
    print(f"  Invoke Total:      {total_invoke:.3f}s")
    print()
    print("Key insight: Streaming does NOT make generation faster overall.")
    print("It makes the PERCEIVED latency much lower because the user sees")
    print("tokens arriving immediately instead of waiting for the full response.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all streaming demonstrations."""
    print("=" * 60)
    print("  LangChain Fundamentals: Streaming Tokens from LLMs")
    print("=" * 60)

    # Load configuration and initialize LLM
    config = load_config()
    llm = get_llm()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Temperature: {config['default_temperature']}")

    # Part 1: Basic streaming with llm.stream()
    demonstrate_basic_streaming(llm)

    # Part 2: Streaming through a chain
    demonstrate_chain_streaming(llm)

    # Part 3: Async streaming with .astream()
    asyncio.run(demonstrate_async_streaming(llm))

    # Part 4: Timing metrics - TTFT vs total time
    demonstrate_timing_metrics(llm)

    print("\n" + "=" * 60)
    print("  Exercise 05 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. .stream() yields tokens incrementally (AIMessageChunk objects)")
    print("  2. StrOutputParser in a chain converts chunks to plain strings")
    print("  3. .astream() provides async iteration for non-blocking streaming")
    print("  4. Time-to-First-Token (TTFT) is much lower with streaming")
    print("  5. Total generation time is similar with or without streaming")
    print("  6. Streaming improves perceived latency and user experience")
    print("  7. Use flush=True when printing streamed tokens to see them immediately")
    print("  8. Async streaming is ideal for web frameworks (FastAPI, etc.)")
    print()


if __name__ == "__main__":
    main()
