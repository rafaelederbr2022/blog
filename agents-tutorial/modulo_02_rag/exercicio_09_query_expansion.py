"""
Module 2 - Exercise 09: Query Expansion for Improved Recall
=============================================================
Learn how to use query expansion to improve retrieval recall in RAG systems.
Query expansion generates multiple variations/reformulations of a user query,
retrieves results for each variation, and merges them to find more relevant
documents that a single query might miss.

Concepts covered:
- Generating query variations using an LLM
- Multi-Query Retriever pattern from LangChain
- Retrieving results for each query variation independently
- Merging and deduplicating results from multiple queries
- Comparing recall between single query and expanded queries
- Using LCEL chains for query generation
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
from langchain.retrievers.multi_query import MultiQueryRetriever

import logging


# ---------------------------------------------------------------------------
# Part 1: Setup - Load Documents and Create Vector Store
# ---------------------------------------------------------------------------

def setup_vectorstore(embeddings):
    """Load documents, split them, and create a ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("PART 1: Setup - Loading Documents and Creating Vector Store")
    print("=" * 60)
    print()
    print("We start by loading our sample documents and creating a vector")
    print("store. This is the same foundation used in previous exercises.")
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
        collection_name="query_expansion_collection",
    )

    print(f"  ChromaDB vector store created with {len(chunks)} vectors")
    return vectorstore


# ---------------------------------------------------------------------------
# Part 2: Manual Query Expansion with LLM
# ---------------------------------------------------------------------------

def demonstrate_manual_query_expansion(vectorstore, llm):
    """Show how to manually expand a query using an LLM and retrieve results."""
    print("\n" + "=" * 60)
    print("PART 2: Manual Query Expansion with LLM")
    print("=" * 60)
    print()
    print("Query expansion works by generating multiple reformulations of")
    print("the original query. Each reformulation captures a different angle")
    print("or phrasing, which helps retrieve documents that might be missed")
    print("by the original query alone.")
    print()
    print("Why does this improve recall?")
    print("  - Different phrasings match different document embeddings")
    print("  - Synonyms and related terms expand the search space")
    print("  - Broader coverage of the topic from multiple perspectives")
    print()

    # --- Define the query expansion prompt ---
    print("2a) Generating query variations with an LLM:")
    print("-" * 40)
    print()

    original_query = "How do cloud services handle security?"

    expansion_prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant that helps improve information retrieval. "
        "Given the following user question, generate 3 different versions of "
        "the question that capture different aspects or use different phrasing. "
        "Each version should help retrieve relevant documents that the original "
        "question might miss.\n\n"
        "Original question: {question}\n\n"
        "Provide exactly 3 alternative questions, one per line. "
        "Do not number them or add any prefix."
    )

    # Build the expansion chain using LCEL
    expansion_chain = expansion_prompt | llm | StrOutputParser()

    print(f"  Original query: '{original_query}'")
    print()
    print("  Generating variations...")
    print()

    # Generate expanded queries
    expanded_text = expansion_chain.invoke({"question": original_query})
    expanded_queries = [q.strip() for q in expanded_text.strip().split("\n") if q.strip()]

    print("  Generated query variations:")
    for i, query in enumerate(expanded_queries, 1):
        print(f"    {i}. {query}")
    print()

    # --- Retrieve results for each query ---
    print("2b) Retrieving results for each query variation:")
    print("-" * 40)
    print()

    all_results = []
    seen_contents = set()

    # Retrieve for original query
    original_results = vectorstore.similarity_search(original_query, k=3)
    print(f"  Original query results: {len(original_results)} documents")
    for doc in original_results:
        content_hash = doc.page_content[:100]
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            all_results.append(doc)

    # Retrieve for each expanded query
    for i, query in enumerate(expanded_queries, 1):
        results = vectorstore.similarity_search(query, k=3)
        new_count = 0
        for doc in results:
            content_hash = doc.page_content[:100]
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_results.append(doc)
                new_count += 1
        print(f"  Variation {i} results: {len(results)} retrieved, {new_count} new unique")

    print()
    print(f"  Total unique documents after expansion: {len(all_results)}")
    print(f"  vs. original query alone: {len(original_results)}")
    print(f"  Improvement: +{len(all_results) - len(original_results)} additional documents")
    print()

    # --- Show the merged results ---
    print("2c) Merged and deduplicated results:")
    print("-" * 40)
    print()

    for i, doc in enumerate(all_results[:6], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] Source: {source}")
        print(f"      '{preview}...'")
        print()

    print("  Query expansion found documents from multiple angles that a")
    print("  single query would have missed. This improves recall significantly.")

    return all_results


# ---------------------------------------------------------------------------
# Part 3: MultiQueryRetriever (LangChain Built-in)
# ---------------------------------------------------------------------------

def demonstrate_multi_query_retriever(vectorstore, llm):
    """Show LangChain's built-in MultiQueryRetriever for query expansion."""
    print("\n" + "=" * 60)
    print("PART 3: MultiQueryRetriever (LangChain Built-in)")
    print("=" * 60)
    print()
    print("LangChain provides MultiQueryRetriever which automates the")
    print("query expansion pattern. It uses an LLM to generate multiple")
    print("perspectives of the query and retrieves documents for each.")
    print()
    print("Advantages of MultiQueryRetriever:")
    print("  - Automatic query generation and deduplication")
    print("  - Integrates seamlessly with LangChain chains")
    print("  - Configurable number of query variations")
    print("  - Works with any base retriever")
    print()

    # --- Create the MultiQueryRetriever ---
    print("3a) Creating MultiQueryRetriever:")
    print("-" * 40)
    print()

    # Set up logging to see the generated queries
    logging.basicConfig()
    logger = logging.getLogger("langchain.retrievers.multi_query")
    logger.setLevel(logging.INFO)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    print("  MultiQueryRetriever created from base retriever + LLM")
    print("  The retriever will automatically:")
    print("    1. Generate multiple query perspectives")
    print("    2. Retrieve documents for each perspective")
    print("    3. Deduplicate and return unique results")
    print()

    # --- Use the MultiQueryRetriever ---
    print("3b) Querying with MultiQueryRetriever:")
    print("-" * 40)
    print()

    query = "What are the risks of artificial intelligence?"
    print(f"  Query: '{query}'")
    print()
    print("  (Check logs above for generated query variations)")
    print()

    # Retrieve documents
    results = multi_query_retriever.invoke(query)

    print(f"  Retrieved {len(results)} unique documents (after deduplication)")
    print()

    for i, doc in enumerate(results[:5], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] Source: {source}")
        print(f"      '{preview}...'")
        print()

    # Reset logging
    logger.setLevel(logging.WARNING)

    # --- Compare with standard retriever ---
    print("3c) Comparing MultiQueryRetriever vs standard retriever:")
    print("-" * 40)
    print()

    standard_results = base_retriever.invoke(query)

    print(f"  Standard retriever: {len(standard_results)} documents")
    print(f"  MultiQuery retriever: {len(results)} documents")
    print()

    # Show unique sources
    standard_sources = set(doc.metadata.get("source", "") for doc in standard_results)
    multi_sources = set(doc.metadata.get("source", "") for doc in results)

    print(f"  Standard retriever sources: {standard_sources}")
    print(f"  MultiQuery retriever sources: {multi_sources}")
    print()
    print("  MultiQueryRetriever typically finds documents from more diverse")
    print("  sources because each query variation may match different documents.")

    return multi_query_retriever


# ---------------------------------------------------------------------------
# Part 4: Query Expansion with Custom Prompt
# ---------------------------------------------------------------------------

def demonstrate_custom_expansion_prompt(vectorstore, llm):
    """Show how to customize the query expansion prompt for better results."""
    print("\n" + "=" * 60)
    print("PART 4: Query Expansion with Custom Prompt")
    print("=" * 60)
    print()
    print("You can customize the expansion prompt to generate more targeted")
    print("query variations. For example, you might want variations that:")
    print("  - Focus on different aspects of the question")
    print("  - Use technical vs. layman terminology")
    print("  - Broaden or narrow the scope")
    print()

    # --- Custom expansion prompt ---
    print("4a) Using a custom expansion prompt:")
    print("-" * 40)
    print()

    custom_prompt = ChatPromptTemplate.from_template(
        "You are an expert at reformulating search queries to maximize "
        "information retrieval. Given the question below, generate 4 "
        "alternative versions:\n"
        "1. A more specific/technical version\n"
        "2. A broader/general version\n"
        "3. A version focusing on practical applications\n"
        "4. A version using different terminology/synonyms\n\n"
        "Original question: {question}\n\n"
        "Provide exactly 4 alternative questions, one per line. "
        "Do not include numbers or prefixes."
    )

    custom_chain = custom_prompt | llm | StrOutputParser()

    query = "How does machine learning work?"
    print(f"  Original query: '{query}'")
    print()

    expanded_text = custom_chain.invoke({"question": query})
    expanded_queries = [q.strip() for q in expanded_text.strip().split("\n") if q.strip()]

    print("  Custom expanded queries:")
    labels = ["Technical", "Broader", "Practical", "Synonyms"]
    for i, q in enumerate(expanded_queries[:4]):
        label = labels[i] if i < len(labels) else f"Var {i+1}"
        print(f"    [{label}] {q}")
    print()

    # --- Retrieve and merge ---
    print("4b) Retrieving with custom expanded queries:")
    print("-" * 40)
    print()

    all_docs = []
    seen = set()

    all_queries = [query] + expanded_queries[:4]
    for q in all_queries:
        results = vectorstore.similarity_search(q, k=2)
        for doc in results:
            doc_id = doc.page_content[:80]
            if doc_id not in seen:
                seen.add(doc_id)
                all_docs.append(doc)

    print(f"  Total unique documents retrieved: {len(all_docs)}")
    print(f"  From {len(all_queries)} query variations (original + 4 expanded)")
    print()

    for i, doc in enumerate(all_docs[:5], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] {source}: '{preview}...'")
    print()

    print("  Custom prompts let you control the TYPE of expansion:")
    print("  - Technical expansion finds domain-specific documents")
    print("  - Broader expansion finds related context")
    print("  - Synonym expansion catches different phrasings")


# ---------------------------------------------------------------------------
# Part 5: Query Expansion in a Full RAG Chain
# ---------------------------------------------------------------------------

def demonstrate_expansion_rag_chain(vectorstore, llm):
    """Build a complete RAG chain that uses query expansion for retrieval."""
    print("\n" + "=" * 60)
    print("PART 5: Query Expansion in a Full RAG Chain")
    print("=" * 60)
    print()
    print("Now we combine query expansion with a full RAG pipeline.")
    print("The chain: expand query -> retrieve for each -> merge -> generate answer")
    print()

    # --- Build the expansion + RAG chain ---
    print("5a) Building the expansion RAG chain:")
    print("-" * 40)
    print()

    # Step 1: Query expansion
    expansion_prompt = ChatPromptTemplate.from_template(
        "Generate 3 different versions of the following question to help "
        "retrieve relevant documents. Return only the questions, one per line.\n\n"
        "Question: {question}\n\n"
        "Alternative questions:"
    )

    # Step 2: RAG answer generation
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question based ONLY on the "
        "provided context. If the context doesn't contain enough information, "
        "say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def expand_and_retrieve(question: str) -> str:
        """Expand the query and retrieve documents for all variations."""
        # Generate expanded queries
        expansion_chain = expansion_prompt | llm | StrOutputParser()
        expanded_text = expansion_chain.invoke({"question": question})
        expanded_queries = [q.strip() for q in expanded_text.strip().split("\n") if q.strip()]

        # Retrieve for all queries (original + expanded)
        all_queries = [question] + expanded_queries[:3]
        all_docs = []
        seen = set()

        for q in all_queries:
            results = vectorstore.similarity_search(q, k=2)
            for doc in results:
                doc_id = doc.page_content[:80]
                if doc_id not in seen:
                    seen.add(doc_id)
                    all_docs.append(doc)

        # Format documents as context
        return "\n\n".join(doc.page_content for doc in all_docs)

    # Build the full chain
    full_chain = (
        {
            "context": lambda x: expand_and_retrieve(x["question"]),
            "question": lambda x: x["question"],
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("  Chain structure:")
    print("    1. Expand query into multiple variations (LLM)")
    print("    2. Retrieve documents for each variation")
    print("    3. Merge and deduplicate retrieved documents")
    print("    4. Format context and generate answer (LLM)")
    print()

    # --- Ask questions ---
    print("5b) Asking questions with the expansion RAG chain:")
    print("-" * 40)
    print()

    questions = [
        "What are the ethical implications of AI and how is it regulated?",
        "How do organizations protect against cyber threats in the cloud?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"  Question {i}: {question}")
        print()

        answer = full_chain.invoke({"question": question})
        print(f"  Answer: {answer}")
        print()
        print("  " + "-" * 50)
        print()

    print("  The expansion RAG chain provides more comprehensive answers")
    print("  because it retrieves context from multiple query perspectives,")
    print("  covering more aspects of the question.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all query expansion demonstrations."""
    print("=" * 60)
    print("  Module 2: Query Expansion for Improved Recall")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Setup vector store
    vectorstore = setup_vectorstore(embeddings)

    # Part 2: Manual query expansion
    demonstrate_manual_query_expansion(vectorstore, llm)

    # Part 3: MultiQueryRetriever
    demonstrate_multi_query_retriever(vectorstore, llm)

    # Part 4: Custom expansion prompt
    demonstrate_custom_expansion_prompt(vectorstore, llm)

    # Part 5: Full RAG chain with expansion
    demonstrate_expansion_rag_chain(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 09 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Query expansion generates multiple reformulations of a query")
    print("  2. Each reformulation retrieves different relevant documents")
    print("  3. Merging results improves recall (finds more relevant docs)")
    print("  4. MultiQueryRetriever automates the expansion pattern")
    print("  5. Custom prompts control the type of expansion (technical, broad, etc.)")
    print("  6. Deduplication prevents returning the same document multiple times")
    print("  7. Expansion is especially useful for ambiguous or broad queries")
    print("  8. The trade-off is more LLM calls and retrieval operations")
    print()


if __name__ == "__main__":
    main()
