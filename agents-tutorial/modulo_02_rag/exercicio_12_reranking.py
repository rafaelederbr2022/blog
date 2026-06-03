"""
Module 2 - Exercise 12: Reranking with LLM-Based Scoring
============================================================
Learn how to improve retrieval precision by reranking initial results using
a more powerful model. This exercise demonstrates the "retrieve then rerank"
pattern where we over-fetch candidates and then re-score them for relevance.

Concepts covered:
- The retrieve-then-rerank pattern for improved precision
- Over-fetching initial candidates (k=10) for broader recall
- LLM-based reranking: using an LLM to score document relevance
- Comparing results before and after reranking
- Understanding precision vs recall trade-offs
- When reranking is most beneficial
"""

import sys
sys.path.append('..')
from config import load_config, get_llm, get_embeddings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import json
import re


# ---------------------------------------------------------------------------
# Part 1: Setup - Load Documents and Create Vector Store
# ---------------------------------------------------------------------------

def setup_vectorstore(embeddings):
    """Load documents, split them, and create a ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("PART 1: Setup - Loading Documents and Creating Vector Store")
    print("=" * 60)
    print()
    print("We create a vector store with smaller chunks to have more")
    print("candidates for retrieval. This simulates a real scenario where")
    print("initial retrieval returns many results of varying relevance.")
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

    # Split into smaller chunks to get more retrieval candidates
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size=300, overlap=50)")
    print("  (Smaller chunks = more candidates for reranking to work with)")
    print()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="reranking_collection",
    )

    print(f"  ChromaDB vector store created with {len(chunks)} vectors")
    return vectorstore


# ---------------------------------------------------------------------------
# Part 2: Initial Retrieval (Over-fetching)
# ---------------------------------------------------------------------------

def demonstrate_initial_retrieval(vectorstore):
    """Show how over-fetching retrieves many candidates for reranking."""
    print("\n" + "=" * 60)
    print("PART 2: Initial Retrieval (Over-fetching)")
    print("=" * 60)
    print()
    print("The reranking pattern starts by over-fetching: retrieving more")
    print("documents than we ultimately need. This ensures high recall")
    print("(we don't miss relevant documents) at the cost of lower precision")
    print("(some retrieved documents may not be very relevant).")
    print()
    print("Strategy:")
    print("  - Retrieve k=10 candidates (over-fetch)")
    print("  - Rerank to find the top 3-5 most relevant")
    print("  - This is better than retrieving only k=3 directly")
    print()

    query = "What are the ethical concerns and regulations around AI?"
    print(f"  Query: '{query}'")
    print()

    # Over-fetch with k=10
    results = vectorstore.similarity_search_with_relevance_scores(query, k=10)

    print(f"  Retrieved {len(results)} candidates (over-fetched with k=10):")
    print()

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i:2d}] Score: {score:.4f} | {source}")
        print(f"         '{preview}...'")
    print()

    print("  Notice: The initial retrieval returns documents with varying")
    print("  relevance. Some are highly relevant, others are tangentially")
    print("  related. Reranking will help us identify the truly relevant ones.")

    return query, [doc for doc, score in results]


# ---------------------------------------------------------------------------
# Part 3: LLM-Based Reranking
# ---------------------------------------------------------------------------

def demonstrate_llm_reranking(query, documents, llm):
    """Rerank documents using an LLM to score relevance."""
    print("\n" + "=" * 60)
    print("PART 3: LLM-Based Reranking")
    print("=" * 60)
    print()
    print("Since cross-encoder models require additional dependencies,")
    print("we demonstrate LLM-based reranking. The LLM reads each document")
    print("and scores its relevance to the query on a scale of 1-10.")
    print()
    print("LLM-based reranking advantages:")
    print("  - Uses the same LLM already available in the pipeline")
    print("  - Can understand nuanced relevance (context, intent)")
    print("  - More flexible than embedding similarity alone")
    print()
    print("LLM-based reranking trade-offs:")
    print("  - Slower than cross-encoder (one LLM call per document)")
    print("  - More expensive (token usage for each scoring)")
    print("  - Best used with a small candidate set (10-20 documents)")
    print()

    # --- Define the scoring prompt ---
    scoring_prompt = ChatPromptTemplate.from_template(
        "You are a relevance scoring assistant. Given a query and a document, "
        "score how relevant the document is to answering the query.\n\n"
        "Score on a scale of 1-10 where:\n"
        "  1-3: Not relevant (different topic or tangentially related)\n"
        "  4-6: Somewhat relevant (related topic but doesn't directly answer)\n"
        "  7-9: Highly relevant (directly addresses the query)\n"
        "  10: Perfect match (exactly what the query is looking for)\n\n"
        "Query: {query}\n\n"
        "Document: {document}\n\n"
        "Respond with ONLY a JSON object in this format:\n"
        '{{"score": <number>, "reason": "<brief explanation>"}}'
    )

    scoring_chain = scoring_prompt | llm | StrOutputParser()

    print(f"  Scoring {len(documents)} documents for query:")
    print(f"  '{query}'")
    print()

    # Score each document
    scored_documents = []
    for i, doc in enumerate(documents, 1):
        # Truncate long documents for scoring efficiency
        doc_text = doc.page_content[:500]

        try:
            response = scoring_chain.invoke({
                "query": query,
                "document": doc_text,
            })

            # Parse the JSON response
            # Handle potential markdown code blocks in response
            cleaned = response.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

            result = json.loads(cleaned)
            score = int(result.get("score", 5))
            reason = result.get("reason", "No reason provided")
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback: try to extract a number from the response
            numbers = re.findall(r'\d+', response)
            score = int(numbers[0]) if numbers else 5
            reason = "Score extracted from response"

        scored_documents.append({
            "document": doc,
            "score": score,
            "reason": reason,
        })

        source = doc.metadata.get("source", "unknown")
        print(f"    [{i:2d}] Score: {score}/10 | {source}")
        print(f"         Reason: {reason}")

    print()

    # Sort by score (descending)
    scored_documents.sort(key=lambda x: x["score"], reverse=True)

    return scored_documents


# ---------------------------------------------------------------------------
# Part 4: Compare Before and After Reranking
# ---------------------------------------------------------------------------

def compare_before_after(query, original_documents, scored_documents):
    """Compare document ordering before and after reranking."""
    print("\n" + "=" * 60)
    print("PART 4: Compare Before and After Reranking")
    print("=" * 60)
    print()
    print("Let's compare the document ordering from the initial retrieval")
    print("(based on embedding similarity) with the reranked ordering")
    print("(based on LLM relevance scoring).")
    print()

    # --- Before reranking (original order) ---
    print("4a) BEFORE reranking (embedding similarity order):")
    print("-" * 40)
    print()

    for i, doc in enumerate(original_documents[:5], 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}] {source}")
        print(f"        '{preview}...'")
    print()

    # --- After reranking (LLM-scored order) ---
    print("4b) AFTER reranking (LLM relevance score order):")
    print("-" * 40)
    print()

    for i, item in enumerate(scored_documents[:5], 1):
        doc = item["document"]
        score = item["score"]
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}] Score: {score}/10 | {source}")
        print(f"        '{preview}...'")
    print()

    # --- Analysis ---
    print("4c) Analysis:")
    print("-" * 40)
    print()

    # Check if ordering changed
    original_previews = [doc.page_content[:50] for doc in original_documents[:5]]
    reranked_previews = [item["document"].page_content[:50] for item in scored_documents[:5]]

    if original_previews != reranked_previews:
        print("  The ordering CHANGED after reranking!")
        print("  The LLM identified more relevant documents that were ranked")
        print("  lower by embedding similarity alone.")
    else:
        print("  The ordering remained similar - embedding similarity and LLM")
        print("  scoring agreed on the most relevant documents.")

    print()

    # Show score distribution
    scores = [item["score"] for item in scored_documents]
    high_relevance = sum(1 for s in scores if s >= 7)
    medium_relevance = sum(1 for s in scores if 4 <= s < 7)
    low_relevance = sum(1 for s in scores if s < 4)

    print(f"  Score distribution across {len(scores)} documents:")
    print(f"    High relevance (7-10): {high_relevance} documents")
    print(f"    Medium relevance (4-6): {medium_relevance} documents")
    print(f"    Low relevance (1-3): {low_relevance} documents")
    print()
    print("  By selecting only high-relevance documents, we improve precision")
    print("  while maintaining the recall benefit of over-fetching.")


# ---------------------------------------------------------------------------
# Part 5: Reranking in a Full RAG Pipeline
# ---------------------------------------------------------------------------

def demonstrate_reranking_rag_pipeline(vectorstore, llm):
    """Show reranking integrated into a complete RAG pipeline."""
    print("\n" + "=" * 60)
    print("PART 5: Reranking in a Full RAG Pipeline")
    print("=" * 60)
    print()
    print("In a production RAG system, reranking sits between retrieval")
    print("and generation:")
    print()
    print("  Query -> Retrieve (k=10) -> Rerank -> Top-3 -> Generate Answer")
    print()
    print("This ensures the LLM receives only the most relevant context,")
    print("reducing noise and improving answer quality.")
    print()

    # --- Define the reranking function ---
    scoring_prompt = ChatPromptTemplate.from_template(
        "Score the relevance of this document to the query (1-10). "
        "Respond with ONLY a number.\n\n"
        "Query: {query}\n"
        "Document: {document}\n\n"
        "Score:"
    )

    scoring_chain = scoring_prompt | llm | StrOutputParser()

    def rerank_documents(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        """Rerank documents using LLM scoring and return top_k."""
        scored = []
        for doc in documents:
            try:
                response = scoring_chain.invoke({
                    "query": query,
                    "document": doc.page_content[:400],
                })
                score = int(re.findall(r'\d+', response.strip())[0])
            except (ValueError, IndexError):
                score = 5  # Default score if parsing fails

            scored.append((doc, score))

        # Sort by score descending and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_k]]

    # --- Build the RAG pipeline with reranking ---
    rag_prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the provided context. "
        "If the context doesn't contain enough information, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    rag_chain = rag_prompt | llm | StrOutputParser()

    # --- Execute the pipeline ---
    query = "What are the main challenges of deploying AI in healthcare?"
    print(f"  Query: '{query}'")
    print()

    # Step 1: Over-fetch
    print("  Step 1: Retrieving candidates (k=10)...")
    candidates = vectorstore.similarity_search(query, k=10)
    print(f"    Retrieved {len(candidates)} candidates")
    print()

    # Step 2: Rerank
    print("  Step 2: Reranking to find top 3...")
    top_docs = rerank_documents(query, candidates, top_k=3)
    print(f"    Selected top {len(top_docs)} documents after reranking")
    print()

    for i, doc in enumerate(top_docs, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"    [{i}] {source}: '{preview}...'")
    print()

    # Step 3: Generate answer
    print("  Step 3: Generating answer from reranked context...")
    print()

    context = "\n\n".join(doc.page_content for doc in top_docs)
    answer = rag_chain.invoke({"context": context, "question": query})

    print(f"  Answer: {answer}")
    print()

    # --- Compare with non-reranked pipeline ---
    print("  Comparison - Answer WITHOUT reranking (using top 3 from initial retrieval):")
    print()

    context_no_rerank = "\n\n".join(doc.page_content for doc in candidates[:3])
    answer_no_rerank = rag_chain.invoke({"context": context_no_rerank, "question": query})

    print(f"  Answer: {answer_no_rerank}")
    print()
    print("  The reranked answer typically provides more focused and accurate")
    print("  information because it uses the most relevant context.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all reranking demonstrations."""
    print("=" * 60)
    print("  Module 2: Reranking with LLM-Based Scoring")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Setup vector store
    vectorstore = setup_vectorstore(embeddings)

    # Part 2: Initial retrieval (over-fetching)
    query, initial_documents = demonstrate_initial_retrieval(vectorstore)

    # Part 3: LLM-based reranking
    scored_documents = demonstrate_llm_reranking(query, initial_documents, llm)

    # Part 4: Compare before and after
    compare_before_after(query, initial_documents, scored_documents)

    # Part 5: Full RAG pipeline with reranking
    demonstrate_reranking_rag_pipeline(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 12 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Reranking improves precision by re-scoring retrieved documents")
    print("  2. Over-fetch first (k=10+) to ensure high recall")
    print("  3. LLM-based reranking scores each document's relevance (1-10)")
    print("  4. Cross-encoders are faster but require additional dependencies")
    print("  5. LLM reranking understands nuanced relevance and context")
    print("  6. The pattern: Retrieve (many) -> Rerank -> Select (few) -> Generate")
    print("  7. Reranking adds latency but significantly improves answer quality")
    print("  8. Best for scenarios where precision matters more than speed")
    print()


if __name__ == "__main__":
    main()
