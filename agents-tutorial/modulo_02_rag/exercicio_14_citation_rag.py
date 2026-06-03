"""
Module 2 - Exercise 14: Citation-Based RAG
============================================
Learn how to build a RAG system that generates answers WITH citations
to source documents. Each claim in the answer references which source
document it came from, improving trustworthiness and verifiability.

Concepts covered:
- Building a RAG system that generates answers with citations
- Using structured output (Pydantic) to enforce citation format
- Referencing source documents with numbered citations [1], [2]
- Generating a references section alongside the answer
- How citations improve trustworthiness and verifiability
"""

import sys
sys.path.append('..')
from config import load_config, get_llm, get_embeddings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Models for Structured Citation Output
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single citation referencing a source document."""
    source_number: int = Field(
        description="The number of the source document being cited (e.g., 1, 2, 3)"
    )
    quote: str = Field(
        description="The relevant quote or paraphrase from the source document"
    )


class CitedAnswer(BaseModel):
    """An answer with inline citations and a references section."""
    answer: str = Field(
        description=(
            "The answer to the question with inline citation numbers "
            "in brackets like [1], [2], etc."
        )
    )
    citations: list[Citation] = Field(
        description="List of citations used in the answer, each referencing a source number"
    )


# ---------------------------------------------------------------------------
# Part 1: Setup - Load Documents and Create Vector Store
# ---------------------------------------------------------------------------

def setup_vectorstore(embeddings):
    """Load documents, split them, and create a ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("PART 1: Setup - Loading Documents and Creating Vector Store")
    print("=" * 60)
    print()
    print("We load and index documents, keeping track of their sources.")
    print("Source tracking is essential for citation-based RAG because")
    print("we need to tell the LLM which document each piece of context")
    print("came from.")
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

    # Split into chunks preserving source metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size=400, overlap=50)")
    print("  Each chunk retains its source metadata for citation tracking.")
    print()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="citation_rag_collection",
    )

    print(f"  ChromaDB vector store created with {len(chunks)} vectors")
    return vectorstore


# ---------------------------------------------------------------------------
# Part 2: The Importance of Citations in RAG
# ---------------------------------------------------------------------------

def explain_citations():
    """Explain why citations matter in RAG systems."""
    print("\n" + "=" * 60)
    print("PART 2: Why Citations Matter in RAG")
    print("=" * 60)
    print()
    print("Standard RAG generates answers from retrieved context but doesn't")
    print("tell the user WHERE the information came from. This creates")
    print("trust issues:")
    print()
    print("  Problems without citations:")
    print("    - Users can't verify claims in the answer")
    print("    - No way to distinguish facts from hallucinations")
    print("    - No accountability for incorrect information")
    print("    - Users must blindly trust the system")
    print()
    print("  Benefits of citation-based RAG:")
    print("    - Each claim is traceable to a source document")
    print("    - Users can verify information independently")
    print("    - Hallucinations are easier to detect (no valid source)")
    print("    - Builds trust and transparency in the system")
    print("    - Enables fact-checking and quality auditing")
    print()
    print("  Our approach:")
    print("    - Number each source document [1], [2], [3]...")
    print("    - Instruct the LLM to cite by number in its answer")
    print("    - Use Pydantic structured output to enforce the format")
    print("    - Generate a references section with source details")
    print()


# ---------------------------------------------------------------------------
# Part 3: Citation RAG with Structured Output
# ---------------------------------------------------------------------------

def demonstrate_citation_rag(vectorstore, llm):
    """Build a citation-based RAG system using structured output."""
    print("\n" + "=" * 60)
    print("PART 3: Citation RAG with Structured Output (Pydantic)")
    print("=" * 60)
    print()
    print("We use .with_structured_output() to force the LLM to return")
    print("a CitedAnswer object with:")
    print("  - answer: text with inline [1], [2] citations")
    print("  - citations: list of Citation objects with source_number and quote")
    print()

    query = "What are the main applications of AI in different industries?"
    print(f"  Query: '{query}'")
    print()

    # Step 1: Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    print(f"  Retrieved {len(docs)} source documents:")
    print()

    # Number each document for citation
    numbered_sources = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        numbered_sources.append({
            "number": i,
            "source": source,
            "content": doc.page_content,
        })
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()

    # Step 2: Build the context with numbered sources
    context_parts = []
    for src in numbered_sources:
        context_parts.append(f"[Source {src['number']}] ({src['source']}):\n{src['content']}")

    context = "\n\n".join(context_parts)

    # Step 3: Create the citation prompt
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful research assistant. Answer the user's question "
         "based ONLY on the provided sources. "
         "IMPORTANT: Include inline citations using bracket notation [1], [2], etc. "
         "to reference which source each piece of information comes from. "
         "Every factual claim must have a citation. "
         "Include a citation entry for each source you reference."),
        ("human",
         "Sources:\n{context}\n\n"
         "Question: {question}\n\n"
         "Provide a comprehensive answer with citations."),
    ])

    # Step 4: Use structured output with Pydantic model
    structured_llm = llm.with_structured_output(CitedAnswer)
    chain = citation_prompt | structured_llm

    print("  Generating cited answer...")
    print()

    result = chain.invoke({"context": context, "question": query})

    # Step 5: Display the cited answer
    print("  " + "=" * 50)
    print("  ANSWER WITH CITATIONS:")
    print("  " + "=" * 50)
    print()
    print(f"  {result.answer}")
    print()

    # Display references section
    print("  " + "-" * 50)
    print("  REFERENCES:")
    print("  " + "-" * 50)
    print()

    for citation in result.citations:
        source_info = next(
            (s for s in numbered_sources if s["number"] == citation.source_number),
            None
        )
        source_name = source_info["source"] if source_info else "Unknown"
        print(f"    [{citation.source_number}] {source_name}")
        print(f"        Quote: \"{citation.quote[:120]}...\"")
        print()

    return result, numbered_sources


# ---------------------------------------------------------------------------
# Part 4: Verifying Citations
# ---------------------------------------------------------------------------

def verify_citations(result, numbered_sources):
    """Verify that citations actually reference content in the sources."""
    print("\n" + "=" * 60)
    print("PART 4: Verifying Citations")
    print("=" * 60)
    print()
    print("An important step in citation-based RAG is verifying that")
    print("the citations actually correspond to content in the sources.")
    print("This helps detect hallucinated citations.")
    print()

    valid_count = 0
    invalid_count = 0

    for citation in result.citations:
        source_info = next(
            (s for s in numbered_sources if s["number"] == citation.source_number),
            None
        )

        if source_info is None:
            print(f"  [{citation.source_number}] INVALID - Source number does not exist")
            invalid_count += 1
            continue

        # Check if the quote roughly matches the source content
        # (Using simple substring check - production systems would use
        # semantic similarity)
        source_content = source_info["content"].lower()
        quote_words = citation.quote.lower().split()[:5]  # First 5 words
        match_found = any(word in source_content for word in quote_words if len(word) > 3)

        if match_found:
            print(f"  [{citation.source_number}] VALID - Content found in source")
            valid_count += 1
        else:
            print(f"  [{citation.source_number}] UNCERTAIN - Could not verify exact match")
            print(f"        (This may still be a valid paraphrase)")
            valid_count += 1  # Give benefit of the doubt for paraphrases

    print()
    print(f"  Verification results:")
    print(f"    Valid/Likely valid: {valid_count}")
    print(f"    Invalid: {invalid_count}")
    print(f"    Total citations: {len(result.citations)}")
    print()


# ---------------------------------------------------------------------------
# Part 5: Comparing RAG With and Without Citations
# ---------------------------------------------------------------------------

def compare_with_without_citations(vectorstore, llm):
    """Compare standard RAG output vs citation-based RAG output."""
    print("\n" + "=" * 60)
    print("PART 5: Comparing RAG With and Without Citations")
    print("=" * 60)
    print()

    query = "What are the security challenges in cloud computing?"
    print(f"  Query: '{query}'")
    print()

    # Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    # --- Standard RAG (no citations) ---
    print("  5a) Standard RAG (no citations):")
    print("  " + "-" * 40)
    print()

    standard_prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    standard_chain = standard_prompt | llm | StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in docs)
    standard_answer = standard_chain.invoke({"context": context, "question": query})

    print(f"  {standard_answer}")
    print()
    print("  -> No way to verify which source each claim comes from!")
    print()

    # --- Citation RAG ---
    print("  5b) Citation-based RAG:")
    print("  " + "-" * 40)
    print()

    # Build numbered context
    numbered_context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        numbered_context_parts.append(
            f"[Source {i}] ({source}):\n{doc.page_content}"
        )

    numbered_context = "\n\n".join(numbered_context_parts)

    citation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Answer based ONLY on the provided sources. "
         "Include inline citations [1], [2], etc. for every factual claim. "
         "Every claim must reference a source."),
        ("human",
         "Sources:\n{context}\n\n"
         "Question: {question}"),
    ])

    structured_llm = llm.with_structured_output(CitedAnswer)
    citation_chain = citation_prompt | structured_llm

    cited_result = citation_chain.invoke({
        "context": numbered_context,
        "question": query,
    })

    print(f"  {cited_result.answer}")
    print()
    print("  References:")
    for citation in cited_result.citations:
        source_info = docs[citation.source_number - 1] if citation.source_number <= len(docs) else None
        source_name = source_info.metadata.get("source", "unknown") if source_info else "Unknown"
        print(f"    [{citation.source_number}] {source_name}: \"{citation.quote[:80]}...\"")
    print()
    print("  -> Every claim is traceable to a specific source!")
    print()


# ---------------------------------------------------------------------------
# Part 6: Advanced Citation Patterns
# ---------------------------------------------------------------------------

def demonstrate_advanced_citations(vectorstore, llm):
    """Show advanced citation patterns for production use."""
    print("\n" + "=" * 60)
    print("PART 6: Advanced Citation Patterns")
    print("=" * 60)
    print()
    print("In production systems, you might want more detailed citations.")
    print("Here we demonstrate a multi-question citation approach where")
    print("each answer section cites its sources independently.")
    print()

    # Multiple questions to demonstrate citation tracking
    questions = [
        "What is machine learning and how does it relate to AI?",
        "What are the main types of cyber attacks?",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    structured_llm = llm.with_structured_output(CitedAnswer)

    for q_idx, question in enumerate(questions, 1):
        print(f"  Question {q_idx}: '{question}'")
        print()

        docs = retriever.invoke(question)

        # Build numbered context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Source {i}] ({source}):\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        citation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer based ONLY on the sources provided. "
             "Use inline citations [1], [2], etc. for every claim."),
            ("human", "Sources:\n{context}\n\nQuestion: {question}"),
        ])

        chain = citation_prompt | structured_llm
        result = chain.invoke({"context": context, "question": question})

        print(f"  Answer: {result.answer}")
        print()
        print("  Sources cited:")
        for citation in result.citations:
            if citation.source_number <= len(docs):
                src = docs[citation.source_number - 1].metadata.get("source", "unknown")
                print(f"    [{citation.source_number}] {src}")
        print()
        print("  " + "-" * 40)
        print()

    print("  Advanced patterns for production:")
    print("    - Track citation frequency to identify key sources")
    print("    - Flag answers with no citations as potentially hallucinated")
    print("    - Use citation verification to score answer reliability")
    print("    - Aggregate citations across multiple queries for analytics")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all citation RAG demonstrations."""
    print("=" * 60)
    print("  Module 2: Citation-Based RAG")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm(temperature=0.0)  # Low temperature for factual accuracy
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")
    print(f"Temperature: 0.0 (for factual accuracy with citations)")

    # Part 1: Setup vector store
    vectorstore = setup_vectorstore(embeddings)

    # Part 2: Explain why citations matter
    explain_citations()

    # Part 3: Citation RAG with structured output
    result, numbered_sources = demonstrate_citation_rag(vectorstore, llm)

    # Part 4: Verify citations
    verify_citations(result, numbered_sources)

    # Part 5: Compare with and without citations
    compare_with_without_citations(vectorstore, llm)

    # Part 6: Advanced citation patterns
    demonstrate_advanced_citations(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 14 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Citation-based RAG traces each claim to its source")
    print("  2. Pydantic models enforce structured citation output")
    print("  3. .with_structured_output() guarantees the response format")
    print("  4. Number sources [1], [2] and instruct the LLM to cite them")
    print("  5. Citations improve trust, verifiability, and transparency")
    print("  6. Verify citations by checking quotes against source content")
    print("  7. Hallucinations are easier to detect with citation tracking")
    print("  8. Production systems should log and audit citation quality")
    print()


if __name__ == "__main__":
    main()
