"""
Module 2 - Exercise 08: Metadata Filtering in RAG
====================================================
Learn how to combine semantic search with metadata filtering to create
more precise retrieval. This exercise demonstrates storing documents with
rich metadata in ChromaDB and using Chroma's where/where_document filters
to narrow down results by topic, category, difficulty level, and more.

Concepts covered:
- Loading documents with rich metadata (topic, category, difficulty_level, year)
- Storing documents in ChromaDB with metadata fields
- Filtering by metadata during retrieval (where filters)
- Filtering by document content (where_document filters)
- Combining semantic search with metadata filters for precise retrieval
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


# ---------------------------------------------------------------------------
# Part 1: Load Documents with Rich Metadata
# ---------------------------------------------------------------------------

def demonstrate_rich_metadata_loading():
    """Load documents and assign rich metadata for filtering."""
    print("\n" + "=" * 60)
    print("PART 1: Loading Documents with Rich Metadata")
    print("=" * 60)
    print()
    print("Metadata filtering allows you to narrow search results BEFORE")
    print("semantic similarity is computed. This is powerful for:")
    print("  - Access control (only show docs user has permission for)")
    print("  - Temporal filtering (only recent documents)")
    print("  - Category filtering (only specific topics)")
    print("  - Difficulty-based filtering (match user expertise level)")
    print()

    # --- Define documents with rich metadata ---
    print("1a) Defining rich metadata for each source:")
    print("-" * 40)
    print()

    sources = [
        {
            "path": "dados/artificial_intelligence.txt",
            "metadata": {
                "topic": "artificial_intelligence",
                "category": "technology",
                "difficulty_level": "intermediate",
                "year": 2024,
                "author": "research_team",
                "tags": "ai,ml,nlp,deep_learning",
            },
        },
        {
            "path": "dados/cloud_computing.txt",
            "metadata": {
                "topic": "cloud_computing",
                "category": "infrastructure",
                "difficulty_level": "beginner",
                "year": 2024,
                "author": "devops_team",
                "tags": "cloud,aws,azure,serverless",
            },
        },
        {
            "path": "dados/cybersecurity.txt",
            "metadata": {
                "topic": "cybersecurity",
                "category": "security",
                "difficulty_level": "advanced",
                "year": 2024,
                "author": "security_team",
                "tags": "security,zero_trust,incident_response",
            },
        },
    ]

    # --- Load and enrich documents ---
    all_documents = []

    for src in sources:
        loader = TextLoader(src["path"], encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata.update(src["metadata"])

        all_documents.extend(docs)
        print(f"  Source: {src['path']}")
        for key, value in src["metadata"].items():
            print(f"    {key}: {value}")
        print()

    print(f"  Total documents loaded: {len(all_documents)}")
    print()
    print("  Rich metadata enables powerful filtering capabilities.")
    print("  ChromaDB stores metadata alongside vectors for fast filtering.")

    return all_documents


# ---------------------------------------------------------------------------
# Part 2: Store in ChromaDB with Metadata
# ---------------------------------------------------------------------------

def demonstrate_chromadb_with_metadata(all_documents, embeddings):
    """Split and store documents in ChromaDB preserving all metadata."""
    print("\n" + "=" * 60)
    print("PART 2: Storing Documents in ChromaDB with Metadata")
    print("=" * 60)
    print()
    print("ChromaDB stores metadata as key-value pairs alongside each vector.")
    print("Supported metadata types: str, int, float, bool.")
    print("This metadata can be used for filtering during retrieval.")
    print()

    # --- Split documents ---
    print("2a) Splitting documents into chunks (metadata is preserved):")
    print("-" * 40)
    print()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(all_documents)

    print(f"  Split {len(all_documents)} documents into {len(chunks)} chunks")
    print()

    # Show that metadata is preserved after splitting
    print("  Metadata preserved on chunks (sample):")
    sample = chunks[0]
    print(f"    Topic: {sample.metadata.get('topic')}")
    print(f"    Category: {sample.metadata.get('category')}")
    print(f"    Difficulty: {sample.metadata.get('difficulty_level')}")
    print(f"    Year: {sample.metadata.get('year')}")
    print()

    # --- Create vector store ---
    print("2b) Creating ChromaDB vector store with metadata:")
    print("-" * 40)
    print()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="metadata_filtering_collection",
    )

    print(f"  Vector store created: {len(chunks)} vectors with metadata")
    print(f"  Each vector has associated metadata fields for filtering")
    print()

    # Show metadata distribution
    topic_counts = {}
    for chunk in chunks:
        topic = chunk.metadata.get("topic", "unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("  Metadata distribution:")
    for topic, count in topic_counts.items():
        print(f"    - {topic}: {count} chunks")

    return vectorstore


# ---------------------------------------------------------------------------
# Part 3: Filtering by Metadata (where filters)
# ---------------------------------------------------------------------------

def demonstrate_metadata_filtering(vectorstore):
    """Show how to filter retrieval results using Chroma's where filters."""
    print("\n" + "=" * 60)
    print("PART 3: Filtering by Metadata (where filters)")
    print("=" * 60)
    print()
    print("Chroma's 'where' filter narrows results by metadata BEFORE")
    print("computing similarity. This is more efficient than post-filtering")
    print("and ensures you get k results from the filtered subset.")
    print()

    query = "What are the best practices for protecting systems?"

    # --- Filter by topic ---
    print("3a) Filter by topic (only cybersecurity documents):")
    print("-" * 40)
    print()
    print(f"  Query: '{query}'")
    print(f"  Filter: topic == 'cybersecurity'")
    print()

    results_security = vectorstore.similarity_search(
        query,
        k=3,
        filter={"topic": "cybersecurity"},
    )

    print(f"  Results (only from cybersecurity):")
    for i, doc in enumerate(results_security, 1):
        topic = doc.metadata.get("topic")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] Topic: {topic}")
        print(f"      '{preview}...'")
        print()

    # --- Filter by category ---
    print("3b) Filter by category (only infrastructure documents):")
    print("-" * 40)
    print()
    print(f"  Query: '{query}'")
    print(f"  Filter: category == 'infrastructure'")
    print()

    results_infra = vectorstore.similarity_search(
        query,
        k=3,
        filter={"category": "infrastructure"},
    )

    print(f"  Results (only from infrastructure):")
    for i, doc in enumerate(results_infra, 1):
        category = doc.metadata.get("category")
        topic = doc.metadata.get("topic")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] Category: {category} | Topic: {topic}")
        print(f"      '{preview}...'")
        print()

    # --- Filter by difficulty level ---
    print("3c) Filter by difficulty level (only advanced documents):")
    print("-" * 40)
    print()

    query_advanced = "What are sophisticated attack methods?"
    print(f"  Query: '{query_advanced}'")
    print(f"  Filter: difficulty_level == 'advanced'")
    print()

    results_advanced = vectorstore.similarity_search(
        query_advanced,
        k=3,
        filter={"difficulty_level": "advanced"},
    )

    print(f"  Results (only advanced difficulty):")
    for i, doc in enumerate(results_advanced, 1):
        difficulty = doc.metadata.get("difficulty_level")
        topic = doc.metadata.get("topic")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] Difficulty: {difficulty} | Topic: {topic}")
        print(f"      '{preview}...'")
        print()

    print("  Metadata filtering ensures results come ONLY from the")
    print("  specified subset, regardless of semantic similarity scores")
    print("  from other documents.")


# ---------------------------------------------------------------------------
# Part 4: Advanced Filters (Operators and Combinations)
# ---------------------------------------------------------------------------

def demonstrate_advanced_filters(vectorstore):
    """Show Chroma's advanced filtering with operators and combinations."""
    print("\n" + "=" * 60)
    print("PART 4: Advanced Filters (Operators and Combinations)")
    print("=" * 60)
    print()
    print("ChromaDB supports advanced filter operators:")
    print("  - $eq: equals (default)")
    print("  - $ne: not equals")
    print("  - $in: value in list")
    print("  - $nin: value not in list")
    print("  - $gt, $gte, $lt, $lte: numeric comparisons")
    print("  - $and, $or: logical combinations")
    print()

    query = "What are the main concepts and best practices?"

    # --- $ne operator (not equals) ---
    print("4a) Using $ne (not equals) - exclude a topic:")
    print("-" * 40)
    print()
    print(f"  Query: '{query}'")
    print(f"  Filter: topic != 'cybersecurity'")
    print()

    results_ne = vectorstore.similarity_search(
        query,
        k=4,
        filter={"topic": {"$ne": "cybersecurity"}},
    )

    print(f"  Results (excluding cybersecurity):")
    for i, doc in enumerate(results_ne, 1):
        topic = doc.metadata.get("topic")
        print(f"  [{i}] Topic: {topic} | '{doc.page_content[:60].replace(chr(10), ' ')}...'")
    print()

    # --- $in operator (value in list) ---
    print("4b) Using $in (value in list) - multiple topics:")
    print("-" * 40)
    print()
    print(f"  Query: '{query}'")
    print(f"  Filter: topic in ['artificial_intelligence', 'cybersecurity']")
    print()

    results_in = vectorstore.similarity_search(
        query,
        k=4,
        filter={"topic": {"$in": ["artificial_intelligence", "cybersecurity"]}},
    )

    print(f"  Results (AI or cybersecurity only):")
    for i, doc in enumerate(results_in, 1):
        topic = doc.metadata.get("topic")
        print(f"  [{i}] Topic: {topic} | '{doc.page_content[:60].replace(chr(10), ' ')}...'")
    print()

    # --- $and operator (combine conditions) ---
    print("4c) Using $and (combine multiple conditions):")
    print("-" * 40)
    print()
    print(f"  Query: '{query}'")
    print(f"  Filter: year == 2024 AND category == 'technology'")
    print()

    results_and = vectorstore.similarity_search(
        query,
        k=3,
        filter={
            "$and": [
                {"year": 2024},
                {"category": "technology"},
            ]
        },
    )

    print(f"  Results (year=2024 AND category=technology):")
    for i, doc in enumerate(results_and, 1):
        topic = doc.metadata.get("topic")
        year = doc.metadata.get("year")
        category = doc.metadata.get("category")
        print(f"  [{i}] Topic: {topic} | Year: {year} | Category: {category}")
        print(f"      '{doc.page_content[:70].replace(chr(10), ' ')}...'")
    print()

    # --- $or operator ---
    print("4d) Using $or (either condition matches):")
    print("-" * 40)
    print()
    print(f"  Query: '{query}'")
    print(f"  Filter: category == 'security' OR difficulty_level == 'beginner'")
    print()

    results_or = vectorstore.similarity_search(
        query,
        k=4,
        filter={
            "$or": [
                {"category": "security"},
                {"difficulty_level": "beginner"},
            ]
        },
    )

    print(f"  Results (security OR beginner):")
    for i, doc in enumerate(results_or, 1):
        topic = doc.metadata.get("topic")
        category = doc.metadata.get("category")
        difficulty = doc.metadata.get("difficulty_level")
        print(f"  [{i}] Topic: {topic} | Category: {category} | Difficulty: {difficulty}")
    print()

    print("  Advanced filters give you fine-grained control over which")
    print("  documents are considered during similarity search.")


# ---------------------------------------------------------------------------
# Part 5: where_document Filters (Content-Based Filtering)
# ---------------------------------------------------------------------------

def demonstrate_where_document_filters(vectorstore):
    """Show Chroma's where_document filters for content-based filtering."""
    print("\n" + "=" * 60)
    print("PART 5: Content-Based Filtering (where_document)")
    print("=" * 60)
    print()
    print("Chroma's 'where_document' filter searches within document content")
    print("BEFORE computing similarity. This combines keyword matching with")
    print("semantic search for more precise retrieval.")
    print()
    print("  Operators:")
    print("  - $contains: document must contain the specified text")
    print("  - $not_contains: document must NOT contain the specified text")
    print()

    # --- $contains filter ---
    print("5a) Using $contains - documents mentioning 'Zero Trust':")
    print("-" * 40)
    print()

    query = "What is the modern approach to network security?"
    print(f"  Query: '{query}'")
    print(f"  where_document: $contains 'Zero Trust'")
    print()

    results_contains = vectorstore.similarity_search(
        query,
        k=3,
        filter={"topic": "cybersecurity"},
    )

    # Use the Chroma collection directly for where_document
    collection = vectorstore._collection
    where_doc_results = collection.query(
        query_texts=[query],
        n_results=3,
        where_document={"$contains": "Zero Trust"},
    )

    print(f"  Results containing 'Zero Trust':")
    if where_doc_results and where_doc_results["documents"]:
        for i, doc_text in enumerate(where_doc_results["documents"][0], 1):
            preview = doc_text[:120].replace("\n", " ")
            print(f"  [{i}] '{preview}...'")
            print()

    # --- $not_contains filter ---
    print("5b) Using $not_contains - exclude documents with specific terms:")
    print("-" * 40)
    print()

    query2 = "What are important security concepts?"
    print(f"  Query: '{query2}'")
    print(f"  where_document: $not_contains 'phishing'")
    print()

    where_doc_results2 = collection.query(
        query_texts=[query2],
        n_results=3,
        where={"topic": "cybersecurity"},
        where_document={"$not_contains": "phishing"},
    )

    print(f"  Results (cybersecurity, excluding 'phishing' mentions):")
    if where_doc_results2 and where_doc_results2["documents"]:
        for i, doc_text in enumerate(where_doc_results2["documents"][0], 1):
            preview = doc_text[:120].replace("\n", " ")
            print(f"  [{i}] '{preview}...'")
            print()

    print("  where_document filters are useful when you need to ensure")
    print("  specific keywords are present or absent in retrieved chunks,")
    print("  combining keyword matching with semantic similarity.")


# ---------------------------------------------------------------------------
# Part 6: Combining Semantic Search with Metadata Filters in RAG
# ---------------------------------------------------------------------------

def demonstrate_filtered_rag(vectorstore, llm):
    """Build a RAG chain that uses metadata filtering for precise retrieval."""
    print("\n" + "=" * 60)
    print("PART 6: Filtered RAG Chain")
    print("=" * 60)
    print()
    print("Combining metadata filters with RAG chains allows you to build")
    print("domain-specific Q&A systems. For example, a user can ask about")
    print("security and only get answers from cybersecurity documents.")
    print()

    # --- Create a filtered retriever ---
    print("6a) Creating a filtered retriever (cybersecurity only):")
    print("-" * 40)
    print()

    filtered_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {"topic": "cybersecurity"},
        },
    )

    print("  Retriever configured with filter: topic == 'cybersecurity'")
    print("  Only cybersecurity documents will be retrieved, regardless")
    print("  of how similar other documents might be to the query.")
    print()

    # --- Build filtered RAG chain ---
    def format_docs(docs):
        """Format retrieved documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = ChatPromptTemplate.from_template(
        "You are a cybersecurity expert assistant. Answer questions based "
        "ONLY on the provided context from cybersecurity documents.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    filtered_rag_chain = (
        {
            "context": filtered_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # --- Ask questions ---
    print("6b) Asking questions with the filtered RAG chain:")
    print("-" * 40)
    print()

    questions = [
        "What is the Zero Trust Architecture approach?",
        "How should organizations handle security incidents?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"  Question {i}: {question}")
        print()

        answer = filtered_rag_chain.invoke(question)
        print(f"  Answer: {answer}")
        print()
        print("  " + "-" * 50)
        print()

    # --- Compare filtered vs unfiltered ---
    print("6c) Comparing filtered vs unfiltered retrieval:")
    print("-" * 40)
    print()

    comparison_query = "What are the main service models?"
    print(f"  Query: '{comparison_query}'")
    print()

    # Unfiltered
    unfiltered_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    unfiltered_results = unfiltered_retriever.invoke(comparison_query)

    print("  Unfiltered results:")
    for i, doc in enumerate(unfiltered_results, 1):
        topic = doc.metadata.get("topic")
        print(f"    [{i}] Topic: {topic}")
    print()

    # Filtered (cybersecurity only)
    filtered_results = filtered_retriever.invoke(comparison_query)

    print("  Filtered results (cybersecurity only):")
    for i, doc in enumerate(filtered_results, 1):
        topic = doc.metadata.get("topic")
        print(f"    [{i}] Topic: {topic}")
    print()

    print("  Notice: The unfiltered retriever finds the most semantically")
    print("  similar results across ALL topics, while the filtered retriever")
    print("  only returns results from the specified topic, even if they")
    print("  are less semantically similar to the query.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all metadata filtering demonstrations."""
    print("=" * 60)
    print("  Module 2: Metadata Filtering in RAG")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Load documents with rich metadata
    all_documents = demonstrate_rich_metadata_loading()

    # Part 2: Store in ChromaDB with metadata
    vectorstore = demonstrate_chromadb_with_metadata(all_documents, embeddings)

    # Part 3: Filter by metadata (where filters)
    demonstrate_metadata_filtering(vectorstore)

    # Part 4: Advanced filters (operators and combinations)
    demonstrate_advanced_filters(vectorstore)

    # Part 5: Content-based filtering (where_document)
    demonstrate_where_document_filters(vectorstore)

    # Part 6: Filtered RAG chain
    demonstrate_filtered_rag(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 08 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Metadata filtering narrows results BEFORE similarity search")
    print("  2. ChromaDB supports: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte")
    print("  3. Logical operators $and/$or combine multiple filter conditions")
    print("  4. where_document filters match on document content (keywords)")
    print("  5. Filtered retrievers ensure domain-specific RAG answers")
    print("  6. Combining filters with semantic search gives precise results")
    print("  7. Metadata filtering is more efficient than post-retrieval filtering")
    print("  8. Rich metadata enables access control, temporal, and topic filtering")
    print()


if __name__ == "__main__":
    main()
