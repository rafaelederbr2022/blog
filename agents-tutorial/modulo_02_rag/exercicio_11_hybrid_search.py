"""
Module 2 - Exercise 11: Hybrid Search (BM25 + Embeddings)
============================================================
Learn how to combine keyword-based retrieval (BM25) with semantic retrieval
(embeddings) to build a hybrid search system that captures both exact keyword
matches AND semantic similarity.

Concepts covered:
- BM25 keyword-based retrieval using term frequency and inverse document frequency
- Embedding-based semantic retrieval using ChromaDB
- EnsembleRetriever for combining multiple retrieval strategies
- Configurable weights between BM25 and semantic search
- Comparing results: BM25 only vs embeddings only vs hybrid
- Understanding when hybrid search outperforms individual methods

Optional dependency:
- rank_bm25: Pure Python implementation of BM25 ranking algorithm.
  Install with: pip install rank_bm25
  Used internally by langchain_community.retrievers.BM25Retriever.
"""

import sys
sys.path.append('..')
from config import load_config, get_llm, get_embeddings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Part 1: Setup - Load and Prepare Documents
# ---------------------------------------------------------------------------

def setup_documents():
    """Load documents and split them into chunks for retrieval."""
    print("\n" + "=" * 60)
    print("PART 1: Setup - Loading and Preparing Documents")
    print("=" * 60)
    print()
    print("We load our sample documents and split them into chunks.")
    print("The same chunks will be used by both BM25 and embedding retrievers,")
    print("allowing a fair comparison between the two approaches.")
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

    return chunks


# ---------------------------------------------------------------------------
# Part 2: BM25 Keyword-Based Retrieval
# ---------------------------------------------------------------------------

def demonstrate_bm25_retrieval(chunks):
    """Show how BM25 retrieval works based on keyword matching."""
    print("\n" + "=" * 60)
    print("PART 2: BM25 Keyword-Based Retrieval")
    print("=" * 60)
    print()
    print("BM25 (Best Matching 25) is a classic information retrieval algorithm")
    print("that ranks documents based on term frequency (TF) and inverse document")
    print("frequency (IDF). It excels at finding documents with exact keyword matches.")
    print()
    print("Strengths of BM25:")
    print("  - Excellent at exact keyword matching")
    print("  - Fast and efficient (no neural network needed)")
    print("  - Works well for specific technical terms and proper nouns")
    print("  - No embedding model required")
    print()
    print("Weaknesses of BM25:")
    print("  - Cannot understand synonyms or paraphrases")
    print("  - Misses semantically related content without exact terms")
    print("  - Sensitive to vocabulary mismatch between query and documents")
    print()

    # Create BM25 retriever from document chunks
    bm25_retriever = BM25Retriever.from_documents(chunks, k=4)

    print("  BM25 retriever created from document chunks")
    print()

    # --- Query 1: Specific keyword query ---
    print("2a) Query with specific keywords:")
    print("-" * 40)
    print()

    query1 = "neural networks deep learning"
    print(f"  Query: '{query1}'")
    print()

    results1 = bm25_retriever.invoke(query1)
    print(f"  BM25 found {len(results1)} results:")
    for i, doc in enumerate(results1, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()

    # --- Query 2: Semantic query (harder for BM25) ---
    print("2b) Query with semantic meaning (challenging for BM25):")
    print("-" * 40)
    print()

    query2 = "protecting digital assets from threats"
    print(f"  Query: '{query2}'")
    print("  (This query uses different words than 'cybersecurity' or 'encryption')")
    print()

    results2 = bm25_retriever.invoke(query2)
    print(f"  BM25 found {len(results2)} results:")
    for i, doc in enumerate(results2, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()
    print("  Notice: BM25 may struggle with this query because it looks for")
    print("  exact terms like 'protecting', 'digital', 'assets', 'threats'")
    print("  rather than understanding the semantic meaning (cybersecurity).")

    return bm25_retriever


# ---------------------------------------------------------------------------
# Part 3: Embedding-Based Semantic Retrieval
# ---------------------------------------------------------------------------

def demonstrate_embedding_retrieval(chunks, embeddings):
    """Show how embedding-based retrieval captures semantic similarity."""
    print("\n" + "=" * 60)
    print("PART 3: Embedding-Based Semantic Retrieval")
    print("=" * 60)
    print()
    print("Embedding-based retrieval converts text into dense vectors and finds")
    print("documents by measuring cosine similarity in the embedding space.")
    print("It understands meaning, synonyms, and paraphrases.")
    print()
    print("Strengths of Embeddings:")
    print("  - Understands semantic meaning and context")
    print("  - Handles synonyms and paraphrases naturally")
    print("  - Works well for conceptual/abstract queries")
    print()
    print("Weaknesses of Embeddings:")
    print("  - May miss exact keyword matches in favor of semantic similarity")
    print("  - Requires an embedding model (cost and latency)")
    print("  - Can sometimes return semantically similar but irrelevant results")
    print()

    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hybrid_search_collection",
    )

    # Create retriever from vector store
    embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print(f"  ChromaDB vector store created with {len(chunks)} vectors")
    print()

    # --- Query 1: Specific keyword query ---
    print("3a) Query with specific keywords:")
    print("-" * 40)
    print()

    query1 = "neural networks deep learning"
    print(f"  Query: '{query1}'")
    print()

    results1 = embedding_retriever.invoke(query1)
    print(f"  Embeddings found {len(results1)} results:")
    for i, doc in enumerate(results1, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()

    # --- Query 2: Semantic query ---
    print("3b) Query with semantic meaning:")
    print("-" * 40)
    print()

    query2 = "protecting digital assets from threats"
    print(f"  Query: '{query2}'")
    print()

    results2 = embedding_retriever.invoke(query2)
    print(f"  Embeddings found {len(results2)} results:")
    for i, doc in enumerate(results2, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()
    print("  Notice: Embeddings understand that 'protecting digital assets from")
    print("  threats' is semantically related to cybersecurity content, even")
    print("  without exact keyword matches.")

    return embedding_retriever, vectorstore


# ---------------------------------------------------------------------------
# Part 4: Hybrid Search with EnsembleRetriever
# ---------------------------------------------------------------------------

def demonstrate_hybrid_search(bm25_retriever, embedding_retriever):
    """Combine BM25 and embedding retrieval using EnsembleRetriever."""
    print("\n" + "=" * 60)
    print("PART 4: Hybrid Search with EnsembleRetriever")
    print("=" * 60)
    print()
    print("Hybrid search combines BM25 and embedding retrieval to get the")
    print("best of both worlds: exact keyword matching AND semantic understanding.")
    print()
    print("EnsembleRetriever merges results from multiple retrievers using")
    print("Reciprocal Rank Fusion (RRF), which combines rankings from each")
    print("retriever into a unified ranking.")
    print()
    print("How RRF works:")
    print("  - Each retriever produces a ranked list of documents")
    print("  - RRF assigns a score: 1/(rank + k) for each document in each list")
    print("  - Scores are weighted by the retriever weights and summed")
    print("  - Documents are re-ranked by their combined score")
    print()

    # --- Create EnsembleRetriever with equal weights ---
    print("4a) Hybrid search with equal weights (0.5 BM25 + 0.5 Embeddings):")
    print("-" * 40)
    print()

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, embedding_retriever],
        weights=[0.5, 0.5],
    )

    print("  EnsembleRetriever created with weights: [0.5, 0.5]")
    print("  This gives equal importance to keyword matches and semantic similarity.")
    print()

    query = "protecting digital assets from threats"
    print(f"  Query: '{query}'")
    print()

    results = hybrid_retriever.invoke(query)
    print(f"  Hybrid search found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()

    # --- Demonstrate configurable weights ---
    print("4b) Hybrid search with BM25-heavy weights (0.7 BM25 + 0.3 Embeddings):")
    print("-" * 40)
    print()

    hybrid_bm25_heavy = EnsembleRetriever(
        retrievers=[bm25_retriever, embedding_retriever],
        weights=[0.7, 0.3],
    )

    print("  Weights: [0.7, 0.3] - favoring keyword matches")
    print(f"  Query: '{query}'")
    print()

    results_bm25_heavy = hybrid_bm25_heavy.invoke(query)
    print(f"  Results ({len(results_bm25_heavy)} documents):")
    for i, doc in enumerate(results_bm25_heavy[:3], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}] {source}: '{preview}...'")
    print()

    print("4c) Hybrid search with Embedding-heavy weights (0.3 BM25 + 0.7 Embeddings):")
    print("-" * 40)
    print()

    hybrid_emb_heavy = EnsembleRetriever(
        retrievers=[bm25_retriever, embedding_retriever],
        weights=[0.3, 0.7],
    )

    print("  Weights: [0.3, 0.7] - favoring semantic similarity")
    print(f"  Query: '{query}'")
    print()

    results_emb_heavy = hybrid_emb_heavy.invoke(query)
    print(f"  Results ({len(results_emb_heavy)} documents):")
    for i, doc in enumerate(results_emb_heavy[:3], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}] {source}: '{preview}...'")
    print()

    print("  Weight tuning guidelines:")
    print("  - Use higher BM25 weight for technical/keyword-specific queries")
    print("  - Use higher embedding weight for conceptual/abstract queries")
    print("  - Equal weights (0.5/0.5) is a good default starting point")

    return hybrid_retriever


# ---------------------------------------------------------------------------
# Part 5: Comparison - BM25 vs Embeddings vs Hybrid
# ---------------------------------------------------------------------------

def compare_retrieval_methods(bm25_retriever, embedding_retriever, hybrid_retriever):
    """Compare all three retrieval methods side by side."""
    print("\n" + "=" * 60)
    print("PART 5: Comparison - BM25 vs Embeddings vs Hybrid")
    print("=" * 60)
    print()
    print("Let's compare all three methods on different types of queries")
    print("to understand when each approach excels.")
    print()

    queries = [
        {
            "query": "machine learning algorithms",
            "type": "Keyword-specific",
            "explanation": "Contains exact technical terms - BM25 should do well",
        },
        {
            "query": "how computers learn from data",
            "type": "Semantic/conceptual",
            "explanation": "Paraphrases 'machine learning' - embeddings should excel",
        },
        {
            "query": "cloud infrastructure security vulnerabilities",
            "type": "Mixed (keywords + concepts)",
            "explanation": "Has keywords AND requires semantic understanding - hybrid wins",
        },
    ]

    for q_info in queries:
        query = q_info["query"]
        print(f"  Query: '{query}'")
        print(f"  Type: {q_info['type']}")
        print(f"  Expectation: {q_info['explanation']}")
        print()

        # BM25 results
        bm25_results = bm25_retriever.invoke(query)
        bm25_sources = [doc.metadata.get("source", "?") for doc in bm25_results]

        # Embedding results
        emb_results = embedding_retriever.invoke(query)
        emb_sources = [doc.metadata.get("source", "?") for doc in emb_results]

        # Hybrid results
        hybrid_results = hybrid_retriever.invoke(query)
        hybrid_sources = [doc.metadata.get("source", "?") for doc in hybrid_results]

        print(f"    BM25 ({len(bm25_results)} results):")
        for i, doc in enumerate(bm25_results[:2], 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"      [{i}] '{preview}...'")

        print(f"    Embeddings ({len(emb_results)} results):")
        for i, doc in enumerate(emb_results[:2], 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"      [{i}] '{preview}...'")

        print(f"    Hybrid ({len(hybrid_results)} results):")
        for i, doc in enumerate(hybrid_results[:2], 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"      [{i}] '{preview}...'")

        print()
        print("    " + "-" * 50)
        print()

    print("  Summary:")
    print("  - BM25 excels with exact keyword queries")
    print("  - Embeddings excel with semantic/conceptual queries")
    print("  - Hybrid captures BOTH, providing the most robust retrieval")
    print("  - Hybrid is especially valuable when query type is unknown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all hybrid search demonstrations."""
    print("=" * 60)
    print("  Module 2: Hybrid Search (BM25 + Embeddings)")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    embeddings = get_embeddings()

    print(f"\nUsing embedding model: text-embedding-3-small")

    # Part 1: Setup documents
    chunks = setup_documents()

    # Part 2: BM25 retrieval
    bm25_retriever = demonstrate_bm25_retrieval(chunks)

    # Part 3: Embedding retrieval
    embedding_retriever, vectorstore = demonstrate_embedding_retrieval(chunks, embeddings)

    # Part 4: Hybrid search
    hybrid_retriever = demonstrate_hybrid_search(bm25_retriever, embedding_retriever)

    # Part 5: Comparison
    compare_retrieval_methods(bm25_retriever, embedding_retriever, hybrid_retriever)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 11 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. BM25 uses term frequency for keyword-based retrieval")
    print("  2. Embeddings capture semantic meaning and handle paraphrases")
    print("  3. Hybrid search combines both via EnsembleRetriever")
    print("  4. Reciprocal Rank Fusion (RRF) merges rankings from each method")
    print("  5. Weights are configurable to favor one method over the other")
    print("  6. Hybrid is most robust when query type is unpredictable")
    print("  7. BM25 needs no embedding model - fast and cost-effective")
    print("  8. The best weight ratio depends on your specific use case")
    print()


if __name__ == "__main__":
    main()
