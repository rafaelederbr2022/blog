"""
Module 1 - Exercise 03: Conversation with Memory
==================================================
Learn how to maintain context between conversation turns using message
history. Compare conversations with and without memory to understand
why state management matters for chatbots and assistants.

Concepts covered:
- Conversation WITHOUT memory (stateless, each call is independent)
- Conversation WITH memory using a manual message list
- Using InMemoryChatMessageHistory from langchain_core.chat_history
- RunnableWithMessageHistory for automatic memory management
- How the LLM remembers previous turns when memory is provided
- Using HumanMessage, AIMessage, SystemMessage from langchain_core.messages
"""

import sys
sys.path.append('..')
from config import load_config, get_llm

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


# ---------------------------------------------------------------------------
# Part 1: Manual message history management using a list of messages
# ---------------------------------------------------------------------------

def demonstrate_manual_memory(llm):
    """Show how to maintain context by manually managing a message list."""
    print("\n" + "=" * 60)
    print("PART 1: Manual Message History Management")
    print("=" * 60)
    print()
    print("The simplest way to add memory is to maintain a list of messages")
    print("and pass the full history to the LLM on each call. The model sees")
    print("all previous turns and can reference them in its responses.")
    print()
    print("This is equivalent to ConversationBufferMemory - it stores the")
    print("FULL conversation history without any summarization or trimming.")
    print()

    # Initialize the conversation history with a system message
    conversation_history = [
        SystemMessage(content="You are a helpful assistant. Keep responses brief (1-2 sentences).")
    ]

    # Turn 1: Introduce ourselves
    print("Turn 1: Introducing ourselves")
    print("-" * 40)
    user_msg1 = "Hi! My name is Alice and I work as a data scientist."
    conversation_history.append(HumanMessage(content=user_msg1))

    response1 = llm.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response1.content))

    print(f"Human: {user_msg1}")
    print(f"AI: {response1.content}")
    print(f"[History size: {len(conversation_history)} messages]")
    print()

    # Turn 2: Ask about our name - WITH previous context
    print("Turn 2: Asking about our name (full history is sent)")
    print("-" * 40)
    user_msg2 = "What is my name and what do I do for work?"
    conversation_history.append(HumanMessage(content=user_msg2))

    response2 = llm.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response2.content))

    print(f"Human: {user_msg2}")
    print(f"AI: {response2.content}")
    print(f"[History size: {len(conversation_history)} messages]")
    print()

    # Turn 3: Ask a follow-up that requires context from both turns
    print("Turn 3: Follow-up requiring multi-turn context")
    print("-" * 40)
    user_msg3 = "Can you suggest a Python library that would be useful for my job?"
    conversation_history.append(HumanMessage(content=user_msg3))

    response3 = llm.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response3.content))

    print(f"Human: {user_msg3}")
    print(f"AI: {response3.content}")
    print(f"[History size: {len(conversation_history)} messages]")
    print()

    print("The LLM remembers Alice is a data scientist and can give")
    print("relevant suggestions based on the full conversation context.")
    print()

    # Show the full conversation history
    print("Full conversation history (buffer memory stores everything):")
    print("-" * 40)
    for i, msg in enumerate(conversation_history):
        role = type(msg).__name__.replace("Message", "")
        print(f"  [{i}] {role}: {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")


# ---------------------------------------------------------------------------
# Part 2: Using ChatMessageHistory and RunnableWithMessageHistory
# ---------------------------------------------------------------------------

def demonstrate_runnable_with_history(llm):
    """Show automatic memory management with RunnableWithMessageHistory."""
    print("\n" + "=" * 60)
    print("PART 2: RunnableWithMessageHistory (Automatic Memory)")
    print("=" * 60)
    print()
    print("RunnableWithMessageHistory wraps a chain and automatically")
    print("manages message history per session. You provide a function")
    print("that returns a ChatMessageHistory for a given session_id,")
    print("and the wrapper handles loading/saving messages automatically.")
    print()
    print("This is the modern LangChain approach to conversation memory.")
    print()

    # Create a prompt template with a placeholder for message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel advisor. Keep responses brief (2-3 sentences). "
                   "Remember the user's preferences throughout the conversation."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Create the chain: prompt -> LLM
    chain = prompt | llm

    # Store for session histories (simulates a session store)
    session_store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        """Retrieve or create a message history for the given session."""
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    # Wrap the chain with message history management
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Configuration for session tracking
    config = {"configurable": {"session_id": "travel_session_1"}}

    # Turn 1: State preferences
    print("Turn 1: Stating travel preferences")
    print("-" * 40)
    input1 = "I love beach destinations and I'm planning a trip for December."
    response1 = chain_with_history.invoke({"input": input1}, config=config)
    print(f"Human: {input1}")
    print(f"AI: {response1.content}")
    print()

    # Turn 2: Ask for recommendations (should remember beach + December)
    print("Turn 2: Asking for recommendations (memory is automatic)")
    print("-" * 40)
    input2 = "What destinations would you recommend for me?"
    response2 = chain_with_history.invoke({"input": input2}, config=config)
    print(f"Human: {input2}")
    print(f"AI: {response2.content}")
    print()

    # Turn 3: Add a constraint (should remember everything)
    print("Turn 3: Adding a budget constraint")
    print("-" * 40)
    input3 = "My budget is around $2000. Which of those fits best?"
    response3 = chain_with_history.invoke({"input": input3}, config=config)
    print(f"Human: {input3}")
    print(f"AI: {response3.content}")
    print()

    # Show what's stored in the session history
    history = get_session_history("travel_session_1")
    print(f"Messages automatically stored in session: {len(history.messages)}")
    print("-" * 40)
    for i, msg in enumerate(history.messages):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{i}] {role}: {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")
    print()

    # Demonstrate session isolation: a different session has no memory
    print("Session isolation: different session_id = fresh memory")
    print("-" * 40)
    config2 = {"configurable": {"session_id": "travel_session_2"}}
    input_new = "What did I say my budget was?"
    response_new = chain_with_history.invoke({"input": input_new}, config=config2)
    print(f"Human (session 2): {input_new}")
    print(f"AI (session 2): {response_new.content}")
    print()
    print("Different sessions are isolated - session 2 has no memory of session 1.")


# ---------------------------------------------------------------------------
# Part 3: Multi-turn conversation where the LLM remembers context
# ---------------------------------------------------------------------------

def demonstrate_multi_turn_memory(llm):
    """Demonstrate a longer multi-turn conversation with persistent context."""
    print("\n" + "=" * 60)
    print("PART 3: Multi-Turn Conversation with Persistent Context")
    print("=" * 60)
    print()
    print("This demonstrates a realistic multi-turn conversation where the")
    print("LLM builds up knowledge about the user over several exchanges.")
    print("Each response is informed by ALL previous turns.")
    print()

    # Use InMemoryChatMessageHistory for structured storage
    history = InMemoryChatMessageHistory()

    system_msg = SystemMessage(
        content="You are a career coach. Keep responses brief (2-3 sentences). "
                "Build on what you learn about the user across the conversation."
    )

    # Define a multi-turn conversation
    turns = [
        "I've been working as a backend developer for 5 years using Java.",
        "I'm interested in transitioning to machine learning. Is that realistic?",
        "I already know Python basics. What should I focus on next?",
        "Given my background, how long do you think the transition would take?",
        "Can you summarize what you know about me and your recommendations?"
    ]

    for i, user_input in enumerate(turns, 1):
        print(f"Turn {i}:")
        print("-" * 40)

        # Add user message to history
        history.add_user_message(user_input)

        # Build full message list: system + history
        messages = [system_msg] + history.messages
        response = llm.invoke(messages)

        # Add AI response to history
        history.add_ai_message(response.content)

        print(f"Human: {user_input}")
        print(f"AI: {response.content}")
        print(f"[Total messages in history: {len(history.messages)}]")
        print()

    print("Notice how the final response summarizes information gathered")
    print("across ALL previous turns - this is the power of conversation memory.")


# ---------------------------------------------------------------------------
# Part 4: What happens without memory (LLM forgets previous turns)
# ---------------------------------------------------------------------------

def demonstrate_without_memory(llm):
    """Show how the LLM forgets everything between calls without memory."""
    print("\n" + "=" * 60)
    print("PART 4: Without Memory - The LLM Forgets Everything")
    print("=" * 60)
    print()
    print("Without memory, each call to the LLM is completely independent.")
    print("The model has no knowledge of previous interactions. This is the")
    print("default behavior - LLMs are stateless by nature.")
    print()

    system_msg = SystemMessage(
        content="You are a helpful assistant. Keep responses brief (1-2 sentences)."
    )

    # First message: introduce ourselves
    print("Turn 1: Introducing ourselves")
    print("-" * 40)
    response1 = llm.invoke([
        system_msg,
        HumanMessage(content="Hi! My name is Bob and I'm a chef specializing in Italian cuisine.")
    ])
    print(f"Human: Hi! My name is Bob and I'm a chef specializing in Italian cuisine.")
    print(f"AI: {response1.content}")
    print()

    # Second message: ask about our name - but WITHOUT previous context
    print("Turn 2: Asking about our name (NO memory - fresh context)")
    print("-" * 40)
    response2 = llm.invoke([
        system_msg,
        HumanMessage(content="What is my name and what do I specialize in?")
    ])
    print(f"Human: What is my name and what do I specialize in?")
    print(f"AI: {response2.content}")
    print()

    print("The LLM cannot answer because it has no memory of Turn 1.")
    print("Each invocation is a completely fresh start.")
    print()

    # Side-by-side comparison
    print("COMPARISON: Same question WITH vs WITHOUT memory")
    print("=" * 40)
    print()

    context_msg = "I'm learning Python and I just finished a course on decorators."
    follow_up = "What topic should I learn next based on what I just told you?"

    print(f"Context: '{context_msg}'")
    print(f"Question: '{follow_up}'")
    print()

    # WITHOUT memory
    print("WITHOUT memory (only the question is sent):")
    print("-" * 40)
    response_no_mem = llm.invoke([
        system_msg,
        HumanMessage(content=follow_up)
    ])
    print(f"AI: {response_no_mem.content}")
    print()

    # WITH memory
    print("WITH memory (context + question are sent together):")
    print("-" * 40)
    response_with_mem = llm.invoke([
        system_msg,
        HumanMessage(content=context_msg),
        AIMessage(content="That's great! Decorators are a powerful Python feature."),
        HumanMessage(content=follow_up)
    ])
    print(f"AI: {response_with_mem.content}")
    print()

    print("Key insight:")
    print("  - Without memory: Generic answer, no awareness of decorators")
    print("  - With memory: Specific answer building on the decorators context")
    print()
    print("Memory is essential for any conversational AI application!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all conversation memory demonstrations."""
    print("=" * 60)
    print("  LangChain Fundamentals: Conversation with Memory")
    print("=" * 60)

    # Load configuration and initialize LLM
    config = load_config()
    llm = get_llm()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Temperature: {config['default_temperature']}")

    # Part 1: Manual message history (ConversationBufferMemory equivalent)
    demonstrate_manual_memory(llm)

    # Part 2: RunnableWithMessageHistory for automatic memory
    demonstrate_runnable_with_history(llm)

    # Part 3: Multi-turn conversation with persistent context
    demonstrate_multi_turn_memory(llm)

    # Part 4: Without memory - LLM forgets
    demonstrate_without_memory(llm)

    print("\n" + "=" * 60)
    print("  Exercise 03 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. LLMs are stateless: each call is independent by default")
    print("  2. Manual message list: append HumanMessage/AIMessage to maintain context")
    print("     (equivalent to ConversationBufferMemory - stores full history)")
    print("  3. InMemoryChatMessageHistory: structured storage from langchain_core")
    print("  4. RunnableWithMessageHistory: automatic memory per session")
    print("  5. Session isolation: different session_ids have separate histories")
    print("  6. Trade-off: more history = more tokens = higher cost and latency")
    print("     (ConversationSummaryMemory addresses this by summarizing instead)")
    print(" Be careful: sending the full history every time increases token usage !!")
    print()


if __name__ == "__main__":
    main()
