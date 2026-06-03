"""
Module 2 - Exercise 06: Basic RAG (Retrieval-Augmented Generation)
====================================================================
Learn how to build a complete RAG pipeline from scratch. RAG enhances LLM
responses by retrieving relevant context from a knowledge base before
generating an answer, reducing hallucinations and grounding responses in facts.

Concepts covered:
- Loading documents from a directory using TextLoader and DirectoryLoader
- Splitting documents into chunks with RecursiveCharacterTextSplitter
- Creating embeddings and storing them in a ChromaDB vector store
- Querying the vector store with similarity search
- Building a complete RAG chain: retrieve context -> format prompt -> LLM -> answer
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
# Part 1: Load and Split Documents
# ---------------------------------------------------------------------------

def demonstrate_loading_and_splitting():
    """Show how to load documents from files and split them into chunks."""
    print("\n" + "=" * 60)
    print("PART 1: Loading and Splitting Documents")
    print("=" * 60)
    print()
    print("RAG starts with loading your knowledge base into manageable chunks.")
    print("We use loaders to read files and splitters to break them into")
    print("smaller pieces that fit within embedding model context windows.")
    print()

    # --- Loading a single document with TextLoader ---
    print("1a) Loading a single document with TextLoader:")
    print("-" * 40)

    single_loader = TextLoader("dados/artificial_intelligence.txt", encoding="utf-8")
    single_doc = single_loader.load()

    print(f"  Loaded 1 file -> {len(single_doc)} document(s)")
    print(f"  Content preview: '{single_doc[0].page_content[:80]}...'")
    print(f"  Metadata: {single_doc[0].metadata}")
    print()

    # --- Loading all documents from a directory with DirectoryLoader ---
    print("1b) Loading all .txt files from a directory with DirectoryLoader:")
    print("-" * 40)

    dir_loader = DirectoryLoader(
        "dados/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = dir_loader.load()

    print(f"  Loaded {len(documents)} documents from dados/ directory")
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        print(f"    - {source} ({len(doc.page_content)} characters)")
    print()

    # --- Splitting documents into chunks ---
    print("1c) Splitting documents into chunks with RecursiveCharacterTextSplitter:")
    print("-" * 40)
    print()
    print("  Why split? Documents are often too large for embedding models and")
    print("  retrieving a full document may include irrelevant information.")
    print("  Smaller chunks allow more precise retrieval.")
    print()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"  Original: {len(documents)} documents")
    print(f"  After splitting: {len(chunks)} chunks")
    print(f"  Chunk size: 500 characters (with 50 character overlap)")
    print()
    print("  Sample chunk (first one):")
    print(f"    Content: '{chunks[0].page_content[:100]}...'")
    print(f"    Metadata: {chunks[0].metadata}")
    print()
    print("  The RecursiveCharacterTextSplitter tries to split at natural")
    print("  boundaries (paragraphs, sentences) before falling back to")
    print("  character-level splits. Overlap ensures context is not lost")
    print("  at chunk boundaries.")

    return chunks


# ---------------------------------------------------------------------------
# Part 2: Create Embeddings and Store in ChromaDB
# ---------------------------------------------------------------------------

def demonstrate_embeddings_and_vectorstore(chunks, embeddings):
    """Show how to create embeddings and store them in a vector database."""
    print("\n" + "=" * 60)
    print("PART 2: Creating Embeddings and Storing in ChromaDB")
    print("=" * 60)
    print()
    print("Embeddings convert text into numerical vectors that capture semantic")
    print("meaning. Similar texts produce similar vectors, enabling semantic search.")
    print("ChromaDB is a lightweight vector database that stores these embeddings.")
    print()

    # --- Create vector store from document chunks ---
    print("2a) Creating ChromaDB vector store from document chunks:")
    print("-" * 40)
    print()
    print("  This step:")
    print("  1. Takes each chunk of text")
    print("  2. Sends it to the embedding model (text-embedding-3-small)")
    print("  3. Stores the vector + original text + metadata in ChromaDB")
    print()

    # Use an in-memory Chroma collection (no persistence needed for demo)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_study_collection",
    )

    print(f"  Vector store created successfully!")
    print(f"  Number of vectors stored: {len(chunks)}")
    print(f"  Embedding model: text-embedding-3-small")
    print(f"  Collection name: 'rag_study_collection'")
    print()

    # --- Show how embeddings work ---
    print("2b) Understanding embeddings:")
    print("-" * 40)
    print()

    sample_text = "What is machine learning?"
    sample_embedding = embeddings.embed_query(sample_text)

    print(f"  Text: '{sample_text}'")
    print(f"  Embedding dimensions: {len(sample_embedding)}")
    print(f"  First 5 values: {sample_embedding[:5]}")
    print()
    print("  Each text is converted to a vector of ~1536 dimensions.")
    print("  Semantically similar texts will have vectors that are close")
    print("  together in this high-dimensional space.")

    return vectorstore


# ---------------------------------------------------------------------------
# Part 3: Query the Vector Store with Similarity Search
# ---------------------------------------------------------------------------

def demonstrate_similarity_search(vectorstore):
    """Show how to query the vector store to find relevant documents."""
    print("\n" + "=" * 60)
    print("PART 3: Querying the Vector Store (Similarity Search)")
    print("=" * 60)
    print()
    print("Similarity search finds the chunks most semantically similar to")
    print("your query. The query is embedded and compared against all stored")
    print("vectors using cosine similarity or other distance metrics.")
    print()

    # --- Basic similarity search ---
    print("3a) Basic similarity search:")
    print("-" * 40)

    query = "What are the main types of cloud computing services?"
    print(f"  Query: '{query}'")
    print()

    results = vectorstore.similarity_search(query, k=3)

    print(f"  Top {len(results)} most relevant chunks:")
    print()
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] Source: {source}")
        print(f"      Content: '{preview}...'")
        print()

    # --- Similarity search with scores ---
    print("3b) Similarity search with relevance scores:")
    print("-" * 40)

    query2 = "How does zero trust security work?"
    print(f"  Query: '{query2}'")
    print()

    results_with_scores = vectorstore.similarity_search_with_score(query2, k=3)

    print(f"  Results with distance scores (lower = more similar):")
    print()
    for i, (doc, score) in enumerate(results_with_scores, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] Score: {score:.4f} | Source: {source}")
        print(f"      Content: '{preview}...'")
        print()

    print("  Note: ChromaDB returns L2 (Euclidean) distance by default.")
    print("  Lower scores indicate higher similarity.")


# ---------------------------------------------------------------------------
# Part 4: Complete RAG Pipeline (Retriever -> Prompt -> LLM -> Answer)
# ---------------------------------------------------------------------------

def demonstrate_rag_chain(vectorstore, llm):
    """Show the complete RAG pipeline combining retrieval with generation."""
    print("\n" + "=" * 60)
    print("PART 4: Complete RAG Pipeline")
    print("=" * 60)
    print()
    print("The full RAG pipeline:")
    print("  1. User asks a question")
    print("  2. Retriever finds relevant document chunks")
    print("  3. Retrieved context is formatted into a prompt")
    print("  4. LLM generates an answer grounded in the context")
    print()
    print("This ensures the LLM answers based on YOUR data, not just its")
    print("training knowledge, reducing hallucinations significantly.")
    print()

    # --- Create a retriever from the vector store ---
    print("4a) Setting up the RAG chain components:")
    print("-" * 40)
    print()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    print("  Retriever: similarity search, top 3 results")
    print()

    # --- Define the RAG prompt template ---
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions based on the "
        "provided context. If the context does not contain enough information "
        "to answer the question, say so clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    print("  Prompt template: includes context placeholder and question")
    print()

    # --- Helper function to format retrieved documents ---
    def format_docs(docs):
        """Join retrieved document contents with double newlines."""
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Build the RAG chain using LCEL ---
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("  Chain: {context: retriever|format, question: passthrough}")
    print("         -> prompt -> LLM -> StrOutputParser")
    print()

    # --- Ask questions using the RAG chain ---
    print("4b) Asking questions with the RAG chain:")
    print("-" * 40)
    print()

    questions = [
        "What are the three service models in cloud computing and what does each provide?",
        "What is the Zero Trust Architecture approach to cybersecurity?",
        "How are Large Language Models used in Natural Language Processing?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"  Question {i}: {question}")
        print()

        answer = rag_chain.invoke(question)

        print(f"  Answer: {answer}")
        print()
        print("  " + "-" * 50)
        print()

    # --- Show what the retriever returns for transparency ---
    print("4c) Inspecting what the retriever provides (transparency):")
    print("-" * 40)
    print()

    inspection_query = "What are ethical concerns about AI?"
    retrieved_docs = retriever.invoke(inspection_query)

    print(f"  Query: '{inspection_query}'")
    print(f"  Retrieved {len(retrieved_docs)} chunks:")
    print()
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] From: {source}")
        print(f"      '{doc.page_content[:150]}...'")
        print()

    print("  These chunks are concatenated and inserted into the prompt")
    print("  as the 'context' that the LLM uses to generate its answer.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function - runs all RAG demonstrations."""
    print("=" * 60)
    print("  Module 2: Basic RAG (Retrieval-Augmented Generation)")
    print("=" * 60)

    # Load configuration and initialize components
    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()

    print(f"\nUsing model: {config['default_model']}")
    print(f"Embedding model: text-embedding-3-small")

    # Part 1: Load and split documents
    chunks = demonstrate_loading_and_splitting()

    # Part 2: Create embeddings and store in ChromaDB
    vectorstore = demonstrate_embeddings_and_vectorstore(chunks, embeddings)

    # Part 3: Query the vector store
    demonstrate_similarity_search(vectorstore)

    # Part 4: Complete RAG pipeline
    demonstrate_rag_chain(vectorstore, llm)

    # Cleanup: delete the in-memory collection
    vectorstore.delete_collection()
    print("\n" + "=" * 60)
    print("  Exercise 06 Complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. RAG = Retrieval + Augmented Generation")
    print("  2. Documents are loaded, split into chunks, and embedded")
    print("  3. ChromaDB stores vectors for fast similarity search")
    print("  4. The retriever finds relevant chunks for a given query")
    print("  5. LCEL chains connect retriever -> prompt -> LLM -> output")
    print("  6. RAG grounds LLM answers in your actual data")
    print("  7. Chunk size and overlap affect retrieval quality")
    print("  8. Similarity search uses vector distance to find relevant content")
    print()


if __name__ == "__main__":
    main()
