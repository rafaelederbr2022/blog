"""
Module 2 - Exercise 10: Query Decomposition for Complex Questions
===================================================================
Learn how to decompose complex, multi-part questions into simpler
sub-questions, retrieve answers for each independently, and combine
them into a comprehensive final answer.

Concepts covered:
- Breaking complex questions into simpler sub-questions using an LLM
- Retrieving context for each sub-question independently
- Answering each sub-question with focused retrieval
- Combining sub-answers into a comprehensive final response
- Comparing decomposed vs. single-query retrieval for complex questions
- Using LCEL chains for decomposition and synthesis
"""

import sys
sys.path.append('..')
from config import load_config, get_llm, get_embeddings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------------------------------------------------------------------------
# Part 1: Setup - Load Documents and Create Vector Store
# ---------------------------------------------------------------------------

def setup_vectorstore(embeddings):
    """Load documents, split them, and create a ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("PART 1: Setup - Loading Documents and Creating Vector Store")
    print("=" * 60)
    print()
    print("Loading our sample documents covering AI, cloud computing, and")
    print("cybersecurity. These multi-topic documents are ideal for testing")
    print("query decomposition since complex questions often span topics.")
    print()

    # Load all documents from the dados/ directory
    dir_loader = DirectoryLoader(
        "dados/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = dir_loader.load()

    print(f"  Loaded {len(documents)} documents from dados/")
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        print(f"    - {source} ({len(doc.page_content)} characters)")
    print()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size=400, overlap=50)")
    print()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="query_decomposition_collection",
    )

    print(f"  ChromaDB vector store created with {len(chunks)} vectors")
    return vectorstore


# ---------------------------------------------------------------------------
# Part 2: Understanding Query Decomposition
# ---------------------------------------------------------------------------

def demonstrate_decomposition_concept(llm):
    """Explain and demonstrate the query decomposition concept."""
    print("\n" + "=" * 60)
    print("PART 2: Understanding Query Decomposition")
    print("=" * 60)
    print()
    print("Complex questions often contain multiple parts that require")
    print("different pieces of information. A single retrieval pass may")
    print("not find all the relevant context for every part.")
    print()
    print("Query decomposition solves this by:")
    print("  1. Breaking the complex question into simpler sub-questions")
    print("  2. Retrieving context for each sub-question independently")
    print("  3. Answering each sub-question with focused context")
    print("  4. Synthesizing sub-answers into a comprehensive response")
    print()
    print("When is decomposition useful?")
    print("  - Multi-part questions ('What is X and how does it relate to Y?')")
    print("  - Comparison questions ('How does A differ from B?')")
    print("  - Questions spanning multiple topics or documents")
    print("  - Questions requiring both factual and analytical answers")
    print()

    # --- Demonstrate decomposition with LLM ---
    print("2a) Decomposing a complex question into sub-questions:")
    print("-" * 40)
    print()

    decomposition_prompt = ChatPromptTemplate.from_template(
        "You are an expert at breaking down complex questions into simpler, "
        "self-contained sub-questions. Each sub-question should be answerable "
        "independently and together they should cover all aspects of the "
        "original question.\n\n"
        "Complex question: {question}\n\n"
        "Break this into 2-4 simpler sub-questions. "
        "Return only the sub-questions, one per line. "
        "Do not number them or add prefixes."
    )

    decomposition_chain = decomposition_prompt | llm | StrOutputParser()

    complex_question = (
        "How do cloud computing services handle security, and what role "
        "does artificial intelligence play in detecting cyber threats?"
    )

    print(f"  Complex question:")
    print(f"    '{complex_question}'")
    print()

    sub_questions_text = decomposition_chain.invoke({"question": complex_question})
    sub_questions = [q.strip() for q in sub_questions_text.strip().split("\n") if q.strip()]

    print("  Decomposed into sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"    {i}. {sq}")
    print()

    print("  Each sub-question targets a specific aspect:")
    print("  - Some focus on cloud security mechanisms")
    print("  - Some focus on AI's role in threat detection")
    print("  - Together they cover the full scope of the original question")

    return sub_questions


# ---------------------------------------------------------------------------
# Part 3: Retrieve and Answer Sub-Questions
# ---------------------------------------------------------------------------

def demonstrate_sub_question_retrieval(vectorstore, llm, sub_questions):
    """Retrieve context and answer each sub-question independently."""
    print("\n" + "=" * 60)
    print("PART 3: Retrieving and Answering Sub-Questions")
    print("=" * 60)
    print()
    print("For each sub-question, we perform independent retrieval and")
    print("generate a focused answer. This ensures each aspect of the")
    print("complex question gets relevant context.")
    print()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    answer_prompt = ChatPromptTemplate.from_template(
        "Answer the following question based ONLY on the provided context. "
        "Be concise and focused.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Answer each sub-question
    sub_answers = []

    for i, sub_q in enumerate(sub_questions, 1):
        print(f"  Sub-question {i}: {sub_q}")
        print("-" * 40)

        # Retrieve relevant documents
        docs = retriever.invoke(sub_q)
        context = format_docs(docs)

        # Generate answer
        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": sub_q})

        sub_answers.append({
            "question": sub_q,
            "answer": answer,
            "num_docs": len(docs),
            "sources": set(doc.metadata.get("source", "unknown") for doc in docs),
        })

        print(f"  Retrieved {len(docs)} documents from: {sub_answers[-1]['sources']}")
        print(f"  Answer: {answer}")
        print()

    print("  Each sub-question retrieved context from potentially different")
    print("  documents, ensuring comprehensive coverage of the topic.")

    return sub_answers


# ---------------------------------------------------------------------------
# Part 4: Synthesize Sub-Answers into Final Response
# ---------------------------------------------------------------------------

def demonstrate_answer_synthesis(llm, original_question, sub_answers):
    """Combine sub-answers into a comprehensive final answer."""
    print("\n" + "=" * 60)
    print("PART 4: Synthesizing Sub-Answers into Final Response")
    print("=" * 60)
    print()
    print("Now we combine all sub-answers into a single, comprehensive")
    print("response that addresses the original complex question fully.")
    print()

    # --- Build synthesis prompt ---
    print("4a) Combining sub-answers:")
    print("-" * 40)
    print()

    # Format sub-answers for the synthesis prompt
    sub_answers_text = ""
    for i, sa in enumerate(sub_answers, 1):
        sub_answers_text += f"Sub-question {i}: {sa['question']}\n"
        sub_answers_text += f"Answer: {sa['answer']}\n\n"

    synthesis_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that synthesizes information from "
        "multiple sources into a comprehensive answer.\n\n"
        "Original question: {original_question}\n\n"
        "The following sub-questions were answered independently:\n\n"
        "{sub_answers}\n\n"
        "Synthesize these answers into a single, coherent, and comprehensive "
        "response to the original question. Make sure to cover all aspects "
        "mentioned in the sub-answers."
    )

    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    print(f"  Original question: '{original_question}'")
    print()
    print("  Sub-answers being synthesized:")
    for i, sa in enumerate(sub_answers, 1):
        print(f"    {i}. [{', '.join(sa['sources'])}] {sa['answer'][:80]}...")
    print()

    # Generate final answer
    final_answer = synthesis_chain.invoke({
        "original_question": original_question,
        "sub_answers": sub_answers_text,
    })

    print("  Final synthesized answer:")
    print(f"  {final_answer}")
    print()

    print("  The synthesized answer combines information from multiple")
    print("  retrieval passes, providing a more complete response than")
    print("  a single retrieval could achieve.")

    return final_answer


# ---------------------------------------------------------------------------
# Part 5: Compare Decomposed vs Single-Query Retrieval
# ---------------------------------------------------------------------------

def demonstrate_comparison(vectorstore, llm):
    """Compare decomposed retrieval with single-query retrieval."""
    print("\n" + "=" * 60)
    print("PART 5: Comparing Decomposed vs Single-Query Retrieval")
    print("=" * 60)
    print()
    print("Let's compare the quality of answers from:")
    print("  A) Single-query retrieval (standard RAG)")
    print("  B) Decomposed retrieval (query decomposition)")
    print()

    complex_question = (
        "What are the main differences between cloud computing deployment "
        "models, and how does each model address security concerns differently?"
    )

    print(f"  Complex question:")
    print(f"    '{complex_question}'")
    print()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Approach A: Single-query retrieval ---
    print("5a) Approach A - Single-query retrieval:")
    print("-" * 40)
    print()

    single_prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    single_docs = retriever.invoke(complex_question)
    single_context = format_docs(single_docs)
    single_sources = set(doc.metadata.get("source", "unknown") for doc in single_docs)

    single_chain = single_prompt | llm | StrOutputParser()
    single_answer = single_chain.invoke({
        "context": single_context,
        "question": complex_question,
    })

    print(f"  Documents retrieved: {len(single_docs)}")
    print(f"  Sources: {single_sources}")
    print(f"  Answer: {single_answer}")
    print()

    # --- Approach B: Decomposed retrieval ---
    print("5b) Approach B - Decomposed retrieval:")
    print("-" * 40)
    print()

    # Decompose
    decomp_prompt = ChatPromptTemplate.from_template(
        "Break the following complex question into 3 simpler sub-questions "
        "that together cover all aspects. Return only the questions, one per line.\n\n"
        "Question: {question}\n\n"
        "Sub-questions:"
    )

    decomp_chain = decomp_prompt | llm | StrOutputParser()
    sub_qs_text = decomp_chain.invoke({"question": complex_question})
    sub_qs = [q.strip() for q in sub_qs_text.strip().split("\n") if q.strip()]

    print(f"  Decomposed into {len(sub_qs)} sub-questions:")
    for i, sq in enumerate(sub_qs, 1):
        print(f"    {i}. {sq}")
    print()

    # Retrieve and answer each
    answer_prompt = ChatPromptTemplate.from_template(
        "Answer concisely based on the context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    all_sources = set()
    all_docs_count = 0
    sub_answers_text = ""

    for sq in sub_qs:
        docs = retriever.invoke(sq)
        all_docs_count += len(docs)
        for doc in docs:
            all_sources.add(doc.metadata.get("source", "unknown"))

        context = format_docs(docs)
        ans_chain = answer_prompt | llm | StrOutputParser()
        ans = ans_chain.invoke({"context": context, "question": sq})
        sub_answers_text += f"Q: {sq}\nA: {ans}\n\n"

    # Synthesize
    synth_prompt = ChatPromptTemplate.from_template(
        "Synthesize these sub-answers into a comprehensive response to "
        "the original question.\n\n"
        "Original question: {question}\n\n"
        "Sub-answers:\n{sub_answers}\n\n"
        "Comprehensive answer:"
    )

    synth_chain = synth_prompt | llm | StrOutputParser()
    decomposed_answer = synth_chain.invoke({
        "question": complex_question,
        "sub_answers": sub_answers_text,
    })

    print(f"  Total documents retrieved: {all_docs_count}")
    print(f"  Sources covered: {all_sources}")
    print(f"  Answer: {decomposed_answer}")
    print()

    # --- Summary ---
    print("5c) Comparison summary:")
    print("-" * 40)
    print()
    print(f"  Single-query approach:")
    print(f"    - Documents: {len(single_docs)}")
    print(f"    - Sources: {len(single_sources)}")
    print(f"    - LLM calls: 1")
    print()
    print(f"  Decomposed approach:")
    print(f"    - Documents: {all_docs_count} (across {len(sub_qs)} sub-queries)")
    print(f"    - Sources: {len(all_sources)}")
    print(f"    - LLM calls: {len(sub_qs) + 2} (decompose + answers + synthesize)")
    print()
    print("  Trade-offs:")
    print("    + Decomposition provides more comprehensive coverage")
    print("    + Each sub-question gets focused, relevant context")
    print("    + Better for multi-part or cross-topic questions")
    print("    - More LLM calls (higher cost and latency)")
    print("    - May introduce inconsistencies between sub-answers")
    print("    - Overkill for simple, single-topic questions")


# ---------------------------------------------------------------------------
# Part 6: Full Decomposition RAG Pipeline
# ---------------------------------------------------------------------------

def demonstrate_full_decomposition_pipeline(vectorstore, llm):
    """Build a complete decomposition RAG pipeline as a reusable function."""
    print("\n" + "=" * 60)
    print("PART 6: Full Decomposition RAG Pipeline")
    print("=" * 60)
    print()
    print("Putting it all together into a reusable pipeline that:")
    print("  1. Detects if a question is complex (multi-part)")
    print("  2. Decomposes it into sub-questions")
    print("  3. Retrieves and answers each sub-question")
    print("  4. Synthesizes a final comprehensive answer")
    print()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def decomposition_rag(question: str) -> str:
        """Complete decomposition RAG pipeline."""
        # Step 1: Decompose the question
        decomp_prompt = ChatPromptTemplate.from_template(
            "Break this question into 2-3 simpler sub-questions that "
            "together fully address the original. Return only the "
            "sub-questions, one per line.\n\n"
            "Question: {question}\n\n"
            "Sub-questions:"
        )
        decomp_chain = decomp_prompt | llm | StrOutputParser()
        sub_qs_text = decomp_chain.invoke({"question": question})
        sub_qs = [q.strip() for q in sub_qs_text.strip().split("\n") if q.strip()]

        # Step 2: Retrieve and answer each sub-question
        sub_answers = []
        for sq in sub_qs:
            docs = retriever.invoke(sq)
            context = "\n\n".join(doc.page_content for doc in docs)

            ans_prompt = ChatPromptTemplate.from_template(
                "Answer concisely based on the context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            ans_chain = ans_prompt | llm | StrOutputParser()
            answer = ans_chain.invoke({"context": context, "question": sq})
            sub_answers.append(f"Q: {sq}\nA: {answer}")

        # Step 3: Synthesize final answer
        synth_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Synthesize the following "
            "sub-answers into a single comprehensive response to the "
            "original question. Be thorough but concise.\n\n"
            "Original question: {question}\n\n"
            "Research findings:\n{findings}\n\n"
            "Comprehensive answer:"
        )
        synth_chain = synth_prompt | llm | StrOutputParser()
        final_answer = synth_chain.invoke({
            "question": question,
            "findings": "\n\n".join(sub_answers),
        })

        return final_answer

    # --- Test with multiple complex questions ---
    print("6a) Testing the decomposition pipeline:")
    print("-" * 40)
    print()

    test_questions = [
        "What are the key challenges in AI ethics and how do they relate to data privacy in cloud computing?",
        "Compare the security approaches used in cloud computing with traditional cybersecurity methods.",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"  Question {i}: {question}")
        print()

        answer = decomposition_rag(question)
        print(f"  Answer: {answer}")
        print()
        print("  " + "-" * 50)
        print()

    print("  The decomposition pipeline handles complex, multi-part questions")
    print("  by breaking them down and addressing each part with focused retrieval.")
    print("  This produces more thorough and accurate answers for complex queries.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all query decomposition demonstrations."""
    print("=" * 60)
    print("  Module 2: Query Decomposition for Complex Questions")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Setup vector store
    vectorstore = setup_vectorstore(embeddings)

    # Part 2: Understanding decomposition
    sub_questions = demonstrate_decomposition_concept(llm)

    # Part 3: Retrieve and answer sub-questions
    original_question = (
        "How do cloud computing services handle security, and what role "
        "does artificial intelligence play in detecting cyber threats?"
    )
    sub_answers = demonstrate_sub_question_retrieval(vectorstore, llm, sub_questions)

    # Part 4: Synthesize final answer
    demonstrate_answer_synthesis(llm, original_question, sub_answers)

    # Part 5: Compare approaches
    demonstrate_comparison(vectorstore, llm)

    # Part 6: Full pipeline
    demonstrate_full_decomposition_pipeline(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 10 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Query decomposition breaks complex questions into sub-questions")
    print("  2. Each sub-question gets independent, focused retrieval")
    print("  3. Sub-answers are synthesized into a comprehensive final response")
    print("  4. Decomposition handles multi-part and cross-topic questions better")
    print("  5. Trade-off: more LLM calls but more thorough answers")
    print("  6. Best for complex questions spanning multiple topics/aspects")
    print("  7. Single-query retrieval is sufficient for simple questions")
    print("  8. Combine with query expansion for maximum recall and coverage")
    print()


if __name__ == "__main__":
    main()
