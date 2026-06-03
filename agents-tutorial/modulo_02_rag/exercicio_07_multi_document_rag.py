"""
Module 2 - Exercise 07: Multi-Document RAG
=============================================
Learn how to build a RAG system that retrieves information from multiple
document sources. This exercise demonstrates loading documents from different
files, enriching them with source metadata, creating a unified vector store,
and querying across all sources while tracking provenance.

Concepts covered:
- Loading documents from multiple sources with distinct metadata
- Adding custom metadata (topic, category, date) to each document
- Creating a unified vector store from heterogeneous sources
- Querying across all documents and identifying result provenance
- Merging results from multiple retrievers using MergerRetriever
"""

import sys
sys.path.append('..')
from config import load_config, get_llm, get_embeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MergerRetriever


# ---------------------------------------------------------------------------
# Part 1: Load Documents from Multiple Sources with Metadata
# ---------------------------------------------------------------------------

def demonstrate_multi_source_loading():
    """Load documents from multiple files and enrich with source metadata."""
    print("\n" + "=" * 60)
    print("PART 1: Loading Documents from Multiple Sources")
    print("=" * 60)
    print()
    print("In real-world RAG systems, knowledge comes from many sources:")
    print("different files, databases, APIs, etc. Each source has its own")
    print("metadata that helps us track where information came from.")
    print()

    # --- Define source metadata for each document ---
    print("1a) Defining source metadata for each document:")
    print("-" * 40)
    print()
    print("  We assign rich metadata to each source so we can later filter")
    print("  and trace which document contributed to each answer.")
    print()

    sources = [
        {
            "path": "dados/artificial_intelligence.txt",
            "metadata": {
                "topic": "artificial_intelligence",
                "category": "technology",
                "date": "2024-01-15",
                "difficulty_level": "intermediate",
            },
        },
        {
            "path": "dados/cloud_computing.txt",
            "metadata": {
                "topic": "cloud_computing",
                "category": "infrastructure",
                "date": "2024-02-20",
                "difficulty_level": "beginner",
            },
        },
        {
            "path": "dados/cybersecurity.txt",
            "metadata": {
                "topic": "cybersecurity",
                "category": "security",
                "date": "2024-03-10",
                "difficulty_level": "advanced",
            },
        },
    ]

    for src in sources:
        print(f"  Source: {src['path']}")
        print(f"    Topic: {src['metadata']['topic']}")
        print(f"    Category: {src['metadata']['category']}")
        print(f"    Date: {src['metadata']['date']}")
        print(f"    Difficulty: {src['metadata']['difficulty_level']}")
        print()

    # --- Load and enrich documents ---
    print("1b) Loading documents and attaching metadata:")
    print("-" * 40)
    print()

    all_documents = []

    for src in sources:
        loader = TextLoader(src["path"], encoding="utf-8")
        docs = loader.load()

        # Enrich each document with custom metadata
        for doc in docs:
            doc.metadata.update(src["metadata"])

        all_documents.extend(docs)
        print(f"  Loaded: {src['path']}")
        print(f"    Documents: {len(docs)}")
        print(f"    Metadata keys: {list(docs[0].metadata.keys())}")
        print()

    print(f"  Total documents loaded: {len(all_documents)}")
    print()
    print("  Each document now carries metadata about its source, topic,")
    print("  category, date, and difficulty level. This metadata travels")
    print("  with the document through splitting and retrieval.")

    return all_documents


# ---------------------------------------------------------------------------
# Part 2: Split and Create Unified Vector Store
# ---------------------------------------------------------------------------

def demonstrate_unified_vectorstore(all_documents, embeddings):
    """Split documents and create a single vector store from all sources."""
    print("\n" + "=" * 60)
    print("PART 2: Creating a Unified Vector Store")
    print("=" * 60)
    print()
    print("All documents from different sources are split into chunks and")
    print("stored in a single vector store. Metadata is preserved on each")
    print("chunk, allowing us to trace results back to their source.")
    print()

    # --- Split all documents ---
    print("2a) Splitting all documents into chunks:")
    print("-" * 40)
    print()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = text_splitter.split_documents(all_documents)

    print(f"  Original documents: {len(all_documents)}")
    print(f"  After splitting: {len(all_chunks)} chunks")
    print()

    # Show chunk distribution by source
    topic_counts = {}
    for chunk in all_chunks:
        topic = chunk.metadata.get("topic", "unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("  Chunks per topic:")
    for topic, count in topic_counts.items():
        print(f"    - {topic}: {count} chunks")
    print()

    # --- Create unified vector store ---
    print("2b) Creating unified ChromaDB vector store:")
    print("-" * 40)
    print()

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="multi_document_collection",
    )

    print(f"  Vector store created with {len(all_chunks)} vectors")
    print(f"  All sources indexed in a single collection")
    print(f"  Metadata preserved for provenance tracking")
    print()
    print("  This unified store allows cross-source semantic search:")
    print("  a single query can find relevant information regardless of")
    print("  which original document it came from.")

    return vectorstore, all_chunks


# ---------------------------------------------------------------------------
# Part 3: Query Across All Documents with Provenance
# ---------------------------------------------------------------------------

def demonstrate_cross_source_query(vectorstore):
    """Query the unified store and show which source each result comes from."""
    print("\n" + "=" * 60)
    print("PART 3: Cross-Source Querying with Provenance")
    print("=" * 60)
    print()
    print("When querying a multi-document store, we can see which source")
    print("each result came from. This is crucial for transparency and")
    print("for building trust in RAG-generated answers.")
    print()

    # --- Query that spans multiple sources ---
    print("3a) Query spanning multiple topics:")
    print("-" * 40)
    print()

    query = "What security measures are important for protecting systems?"
    print(f"  Query: '{query}'")
    print()
    print("  This query should retrieve results from both cybersecurity")
    print("  and cloud computing documents (shared responsibility model).")
    print()

    results = vectorstore.similarity_search_with_score(query, k=5)

    print(f"  Top {len(results)} results:")
    print()
    for i, (doc, score) in enumerate(results, 1):
        topic = doc.metadata.get("topic", "unknown")
        category = doc.metadata.get("category", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] Topic: {topic} | Category: {category} | Score: {score:.4f}")
        print(f"      '{preview}...'")
        print()

    # --- Topic-specific query ---
    print("3b) Query targeting a specific domain:")
    print("-" * 40)
    print()

    query2 = "How do Large Language Models work in NLP?"
    print(f"  Query: '{query2}'")
    print()

    results2 = vectorstore.similarity_search_with_score(query2, k=3)

    print(f"  Top {len(results2)} results:")
    print()
    for i, (doc, score) in enumerate(results2, 1):
        topic = doc.metadata.get("topic", "unknown")
        date = doc.metadata.get("date", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] Topic: {topic} | Date: {date} | Score: {score:.4f}")
        print(f"      '{preview}...'")
        print()

    print("  Notice how the system correctly identifies the most relevant")
    print("  source for each query, even when multiple sources exist.")


# ---------------------------------------------------------------------------
# Part 4: Merging Results from Multiple Retrievers
# ---------------------------------------------------------------------------

def demonstrate_merger_retriever(all_chunks, embeddings):
    """Show how to merge results from multiple independent retrievers."""
    print("\n" + "=" * 60)
    print("PART 4: Merging Results from Multiple Retrievers")
    print("=" * 60)
    print()
    print("Sometimes you want separate vector stores per source (e.g., for")
    print("access control or different update schedules). MergerRetriever")
    print("combines results from multiple retrievers into a single result set.")
    print()

    # --- Create separate vector stores per topic ---
    print("4a) Creating separate vector stores per topic:")
    print("-" * 40)
    print()

    # Split chunks by topic
    ai_chunks = [c for c in all_chunks if c.metadata.get("topic") == "artificial_intelligence"]
    cloud_chunks = [c for c in all_chunks if c.metadata.get("topic") == "cloud_computing"]
    security_chunks = [c for c in all_chunks if c.metadata.get("topic") == "cybersecurity"]

    print(f"  AI chunks: {len(ai_chunks)}")
    print(f"  Cloud chunks: {len(cloud_chunks)}")
    print(f"  Security chunks: {len(security_chunks)}")
    print()

    # Create individual vector stores
    ai_store = Chroma.from_documents(
        documents=ai_chunks,
        embedding=embeddings,
        collection_name="ai_collection",
    )
    cloud_store = Chroma.from_documents(
        documents=cloud_chunks,
        embedding=embeddings,
        collection_name="cloud_collection",
    )
    security_store = Chroma.from_documents(
        documents=security_chunks,
        embedding=embeddings,
        collection_name="security_collection",
    )

    print("  Created 3 separate vector stores (one per topic)")
    print()

    # --- Create retrievers and merge them ---
    print("4b) Creating MergerRetriever from multiple retrievers:")
    print("-" * 40)
    print()

    ai_retriever = ai_store.as_retriever(search_kwargs={"k": 2})
    cloud_retriever = cloud_store.as_retriever(search_kwargs={"k": 2})
    security_retriever = security_store.as_retriever(search_kwargs={"k": 2})

    # MergerRetriever combines results from all retrievers
    merger_retriever = MergerRetriever(
        retrievers=[ai_retriever, cloud_retriever, security_retriever]
    )

    print("  MergerRetriever combines results from:")
    print("    - AI retriever (top 2)")
    print("    - Cloud retriever (top 2)")
    print("    - Security retriever (top 2)")
    print()
    print("  This ensures representation from all sources in the results.")
    print()

    # --- Query the merged retriever ---
    print("4c) Querying the merged retriever:")
    print("-" * 40)
    print()

    query = "What are the best practices for protecting data and systems?"
    print(f"  Query: '{query}'")
    print()

    merged_results = merger_retriever.invoke(query)

    print(f"  Merged results: {len(merged_results)} documents")
    print()
    for i, doc in enumerate(merged_results, 1):
        topic = doc.metadata.get("topic", "unknown")
        category = doc.metadata.get("category", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] Topic: {topic} | Category: {category}")
        print(f"      '{preview}...'")
        print()

    print("  The MergerRetriever guarantees diversity in results by pulling")
    print("  from each source independently, then combining them.")

    # Cleanup
    ai_store.delete_collection()
    cloud_store.delete_collection()
    security_store.delete_collection()

    return merger_retriever


# ---------------------------------------------------------------------------
# Part 5: RAG Chain with Source Attribution
# ---------------------------------------------------------------------------

def demonstrate_rag_with_attribution(vectorstore, llm):
    """Build a RAG chain that includes source attribution in answers."""
    print("\n" + "=" * 60)
    print("PART 5: RAG Chain with Source Attribution")
    print("=" * 60)
    print()
    print("A production RAG system should tell users WHERE the information")
    print("came from. We modify the RAG chain to include source metadata")
    print("in the context, so the LLM can cite its sources.")
    print()

    # --- Create retriever ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # --- Format documents with source information ---
    def format_docs_with_sources(docs):
        """Format documents including their source metadata."""
        formatted = []
        for doc in docs:
            topic = doc.metadata.get("topic", "unknown")
            category = doc.metadata.get("category", "unknown")
            source_info = f"[Source: {topic} | Category: {category}]"
            formatted.append(f"{source_info}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # --- RAG prompt with source attribution instruction ---
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions based on the "
        "provided context. Each piece of context is labeled with its source. "
        "Include source attribution in your answer when citing specific facts.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (cite sources where applicable):"
    )

    # --- Build the chain ---
    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("5a) Asking a cross-source question with attribution:")
    print("-" * 40)
    print()

    question = "What security approaches are relevant for both cloud and traditional systems?"
    print(f"  Question: {question}")
    print()

    answer = rag_chain.invoke(question)
    print(f"  Answer: {answer}")
    print()

    # --- Show retrieved sources for transparency ---
    print("5b) Sources used for the answer above:")
    print("-" * 40)
    print()

    retrieved_docs = retriever.invoke(question)
    for i, doc in enumerate(retrieved_docs, 1):
        topic = doc.metadata.get("topic", "unknown")
        category = doc.metadata.get("category", "unknown")
        print(f"  [{i}] Topic: {topic} | Category: {category}")
        print(f"      '{doc.page_content[:80].replace(chr(10), ' ')}...'")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all multi-document RAG demonstrations."""
    print("=" * 60)
    print("  Module 2: Multi-Document RAG")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Load documents from multiple sources
    all_documents = demonstrate_multi_source_loading()

    # Part 2: Create unified vector store
    vectorstore, all_chunks = demonstrate_unified_vectorstore(all_documents, embeddings)

    # Part 3: Cross-source querying with provenance
    demonstrate_cross_source_query(vectorstore)

    # Part 4: Merge results from multiple retrievers
    demonstrate_merger_retriever(all_chunks, embeddings)

    # Part 5: RAG chain with source attribution
    demonstrate_rag_with_attribution(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 07 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Multi-document RAG loads from multiple sources into one store")
    print("  2. Custom metadata (topic, category, date) enables provenance tracking")
    print("  3. A unified vector store allows cross-source semantic search")
    print("  4. MergerRetriever combines results from independent retrievers")
    print("  5. Source attribution in RAG answers builds user trust")
    print("  6. Metadata travels with chunks through splitting and retrieval")
    print("  7. Different retriever strategies suit different use cases")
    print()


if __name__ == "__main__":
    main()
