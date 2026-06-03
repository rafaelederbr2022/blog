"""
Module 2 - Exercise 13: Context Compression
============================================
Learn how to compress retrieved context before sending it to the LLM.
Retrieved chunks often contain irrelevant information that wastes tokens
and can confuse the model. Context compression extracts only the relevant
portions from each retrieved document.

Concepts covered:
- The problem: retrieved chunks contain irrelevant information
- LLM-based context compression to extract relevant parts
- ContextualCompressionRetriever with LLMChainExtractor
- Before/after comparison: full chunks vs compressed chunks
- How compression reduces token usage while maintaining answer quality
"""

import sys
sys.path.append('..')
from config import load_config, get_llm, get_embeddings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# ---------------------------------------------------------------------------
# Part 1: Setup - Load Documents and Create Vector Store
# ---------------------------------------------------------------------------

def setup_vectorstore(embeddings):
    """Load documents, split them, and create a ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("PART 1: Setup - Loading Documents and Creating Vector Store")
    print("=" * 60)
    print()
    print("We load documents and split them into chunks. These chunks often")
    print("contain a mix of relevant and irrelevant information for any")
    print("given query. Context compression helps solve this problem.")
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

    # Split into chunks - using larger chunks to demonstrate the problem
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size=500, overlap=50)")
    print("  (Larger chunks = more irrelevant content mixed in)")
    print()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="context_compression_collection",
    )

    print(f"  ChromaDB vector store created with {len(chunks)} vectors")
    return vectorstore


# ---------------------------------------------------------------------------
# Part 2: The Problem - Retrieved Chunks Contain Irrelevant Information
# ---------------------------------------------------------------------------

def demonstrate_the_problem(vectorstore):
    """Show how retrieved chunks contain irrelevant information."""
    print("\n" + "=" * 60)
    print("PART 2: The Problem - Irrelevant Information in Retrieved Chunks")
    print("=" * 60)
    print()
    print("When we retrieve chunks from a vector store, each chunk may")
    print("contain sentences that are relevant to the query AND sentences")
    print("that are completely unrelated. This wastes tokens and can")
    print("confuse the LLM when generating answers.")
    print()

    query = "What are the ethical concerns around artificial intelligence?"
    print(f"  Query: '{query}'")
    print()

    # Retrieve documents without compression
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    results = retriever.invoke(query)

    print(f"  Retrieved {len(results)} chunks (WITHOUT compression):")
    print()

    total_chars = 0
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content
        total_chars += len(content)
        print(f"  --- Chunk {i} ({source}) [{len(content)} chars] ---")
        print(f"  {content}")
        print()

    print(f"  Total characters in retrieved context: {total_chars}")
    print()
    print("  PROBLEM: Notice how each chunk contains a mix of relevant")
    print("  and irrelevant sentences. The LLM must process ALL of this")
    print("  text, wasting tokens on information that doesn't help answer")
    print("  the question about ethical concerns.")
    print()

    return query, results


# ---------------------------------------------------------------------------
# Part 3: Context Compression with LLMChainExtractor
# ---------------------------------------------------------------------------

def demonstrate_context_compression(vectorstore, llm, query):
    """Use ContextualCompressionRetriever with LLMChainExtractor."""
    print("\n" + "=" * 60)
    print("PART 3: Context Compression with LLMChainExtractor")
    print("=" * 60)
    print()
    print("ContextualCompressionRetriever wraps a base retriever and")
    print("applies a compressor to each retrieved document. The")
    print("LLMChainExtractor uses an LLM to extract ONLY the parts")
    print("of each document that are relevant to the query.")
    print()
    print("How it works:")
    print("  1. Base retriever fetches documents normally")
    print("  2. LLMChainExtractor processes each document")
    print("  3. The LLM extracts only relevant sentences/passages")
    print("  4. Documents with no relevant content are filtered out")
    print()

    # Create the compressor using LLMChainExtractor
    compressor = LLMChainExtractor.from_llm(llm)

    # Create the compression retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    print(f"  Query: '{query}'")
    print()
    print("  Retrieving with compression...")
    print()

    # Retrieve with compression
    compressed_results = compression_retriever.invoke(query)

    print(f"  Retrieved {len(compressed_results)} compressed chunks:")
    print()

    total_chars_compressed = 0
    for i, doc in enumerate(compressed_results, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content
        total_chars_compressed += len(content)
        print(f"  --- Compressed Chunk {i} ({source}) [{len(content)} chars] ---")
        print(f"  {content}")
        print()

    print(f"  Total characters in compressed context: {total_chars_compressed}")
    print()

    return compressed_results, total_chars_compressed


# ---------------------------------------------------------------------------
# Part 4: Before/After Comparison
# ---------------------------------------------------------------------------

def compare_before_after(query, original_results, compressed_results, total_compressed):
    """Compare full chunks vs compressed chunks."""
    print("\n" + "=" * 60)
    print("PART 4: Before/After Comparison")
    print("=" * 60)
    print()

    total_original = sum(len(doc.page_content) for doc in original_results)

    print("  BEFORE compression:")
    print(f"    - Number of chunks: {len(original_results)}")
    print(f"    - Total characters: {total_original}")
    print(f"    - Average chunk size: {total_original // max(len(original_results), 1)} chars")
    print()

    print("  AFTER compression:")
    print(f"    - Number of chunks: {len(compressed_results)}")
    print(f"    - Total characters: {total_compressed}")
    if compressed_results:
        print(f"    - Average chunk size: {total_compressed // len(compressed_results)} chars")
    print()

    if total_original > 0:
        reduction = ((total_original - total_compressed) / total_original) * 100
        print(f"  REDUCTION: {reduction:.1f}% fewer characters sent to the LLM")
        print(f"  ({total_original} -> {total_compressed} characters)")
    print()

    print("  Benefits of compression:")
    print("    1. Fewer tokens = lower cost per query")
    print("    2. Less noise = more focused answers")
    print("    3. Fits more relevant info in the context window")
    print("    4. Reduces hallucination from irrelevant context")
    print()


# ---------------------------------------------------------------------------
# Part 5: Compression in a Full RAG Pipeline
# ---------------------------------------------------------------------------

def demonstrate_compression_rag_pipeline(vectorstore, llm):
    """Show compression integrated into a complete RAG pipeline."""
    print("\n" + "=" * 60)
    print("PART 5: Compression in a Full RAG Pipeline")
    print("=" * 60)
    print()
    print("Let's compare answer quality with and without compression.")
    print("We'll ask the same question and compare the responses.")
    print()

    query = "How is AI being used in healthcare applications?"
    print(f"  Query: '{query}'")
    print()

    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the provided context. "
        "Be concise and specific.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    rag_chain = rag_prompt | llm | StrOutputParser()

    # --- Without compression ---
    print("  5a) Answer WITHOUT compression:")
    print("  " + "-" * 40)
    print()

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    full_docs = base_retriever.invoke(query)
    full_context = "\n\n".join(doc.page_content for doc in full_docs)

    full_answer = rag_chain.invoke({"context": full_context, "question": query})
    print(f"  Context size: {len(full_context)} characters")
    print(f"  Answer: {full_answer}")
    print()

    # --- With compression ---
    print("  5b) Answer WITH compression:")
    print("  " + "-" * 40)
    print()

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    compressed_docs = compression_retriever.invoke(query)
    compressed_context = "\n\n".join(doc.page_content for doc in compressed_docs)

    compressed_answer = rag_chain.invoke({"context": compressed_context, "question": query})
    print(f"  Context size: {len(compressed_context)} characters")
    print(f"  Answer: {compressed_answer}")
    print()

    # --- Comparison ---
    print("  5c) Comparison:")
    print("  " + "-" * 40)
    print()

    if len(full_context) > 0:
        savings = ((len(full_context) - len(compressed_context)) / len(full_context)) * 100
        print(f"  Token savings: ~{savings:.1f}% reduction in context size")
    print(f"  Full context: {len(full_context)} chars -> Compressed: {len(compressed_context)} chars")
    print()
    print("  The compressed answer should be equally or more accurate because")
    print("  the LLM receives only relevant information, reducing noise.")
    print()


# ---------------------------------------------------------------------------
# Part 6: Manual Compression Approach
# ---------------------------------------------------------------------------

def demonstrate_manual_compression(vectorstore, llm):
    """Show a manual compression approach using a custom LLM chain."""
    print("\n" + "=" * 60)
    print("PART 6: Manual Compression Approach")
    print("=" * 60)
    print()
    print("You can also implement compression manually with a custom prompt.")
    print("This gives you more control over what gets extracted and how.")
    print()

    query = "What security threats exist in cloud computing?"
    print(f"  Query: '{query}'")
    print()

    # Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    # Manual compression prompt
    compression_prompt = ChatPromptTemplate.from_template(
        "Given the following document and a user question, extract ONLY "
        "the sentences or phrases that are directly relevant to answering "
        "the question. If nothing is relevant, respond with 'NO_RELEVANT_CONTENT'.\n\n"
        "Question: {question}\n\n"
        "Document:\n{document}\n\n"
        "Relevant extract:"
    )

    compression_chain = compression_prompt | llm | StrOutputParser()

    print("  Manually compressing each retrieved chunk:")
    print()

    compressed_texts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  --- Chunk {i} ({source}) ---")
        print(f"  Original ({len(doc.page_content)} chars):")
        print(f"    '{doc.page_content[:150]}...'")
        print()

        # Compress using the LLM
        compressed = compression_chain.invoke({
            "question": query,
            "document": doc.page_content,
        })

        if "NO_RELEVANT_CONTENT" not in compressed:
            compressed_texts.append(compressed)
            print(f"  Compressed ({len(compressed)} chars):")
            print(f"    '{compressed[:150]}...'")
        else:
            print("  -> Filtered out (no relevant content)")
        print()

    print(f"  Summary: {len(docs)} chunks -> {len(compressed_texts)} relevant extracts")
    print()
    print("  Manual compression gives you full control over:")
    print("    - The extraction prompt (what counts as 'relevant')")
    print("    - Filtering logic (threshold for relevance)")
    print("    - Output format (sentences, bullet points, etc.)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all context compression demonstrations."""
    print("=" * 60)
    print("  Module 2: Context Compression")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Setup vector store
    vectorstore = setup_vectorstore(embeddings)

    # Part 2: Demonstrate the problem
    query, original_results = demonstrate_the_problem(vectorstore)

    # Part 3: Context compression with LLMChainExtractor
    compressed_results, total_compressed = demonstrate_context_compression(
        vectorstore, llm, query
    )

    # Part 4: Before/after comparison
    compare_before_after(query, original_results, compressed_results, total_compressed)

    # Part 5: Full RAG pipeline comparison
    demonstrate_compression_rag_pipeline(vectorstore, llm)

    # Part 6: Manual compression approach
    demonstrate_manual_compression(vectorstore, llm)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("  Exercise 13 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Retrieved chunks often contain irrelevant information")
    print("  2. Context compression extracts only relevant portions")
    print("  3. LLMChainExtractor uses an LLM to identify relevant content")
    print("  4. ContextualCompressionRetriever wraps any base retriever")
    print("  5. Compression reduces token usage (cost) significantly")
    print("  6. Answer quality is maintained or improved with less noise")
    print("  7. Trade-off: extra LLM call per chunk adds latency")
    print("  8. Manual compression gives more control over extraction")
    print()


if __name__ == "__main__":
    main()
